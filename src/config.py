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
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


class CameraConfig:
    """Camera configuration settings."""
    
    # Auto-detect: Use webcam if RTSP_URL is not set
    _rtsp_url_env = os.getenv("RTSP_URL", "")
    RTSP_URL: str = _rtsp_url_env if _rtsp_url_env else "0"  # Default to webcam index 0
    
    WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    CAMERA_NAME: str = os.getenv("CAMERA_NAME", "Main Camera")
    ROI_SIZE: Tuple[int, int] = (1280, 720) # Match default resolution, no cropping
    
    @classmethod
    def validate(cls) -> bool:
        """Validate camera configuration."""
        if not cls.RTSP_URL:
            logger.warning("RTSP_URL is not set, using default webcam (index 0)")
            cls.RTSP_URL = "0"
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
    TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    COOLDOWN_SECONDS: int = int(os.getenv("DETECTION_COOLDOWN", "900")) # Default 15 minutes
    MAX_FACES: int = int(os.getenv("MAX_FACES", "100")) 
    CAPTURE_MAX_WIDTH: int = int(os.getenv("CAPTURE_MAX_WIDTH", "640"))


class QdrantConfig:
    """Qdrant Vector DB configuration."""
    
    URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    LOCATION: str = os.getenv("QDRANT_LOCATION", ":memory:") # Default to memory for easy testing
    COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "face_attendance")


class AuthServiceConfig:
    """Authentication Service configuration."""
    
    BASE_URL: str = os.getenv("AUTH_SERVICE_URL", "https://auth.bittechx.cloud")
    API_KEY: str = os.getenv("GROUP_AUTH_KEY", "dwX1S5cHAPDYo6Gom2fv8F3D7rNZqPu")
    SYSTEM_ID: str = os.getenv("SYSTEM_ID", "attendance_system")
    SYSTEM_REGISTER: str = os.getenv("SYSTEM_REGISTER", "")


class ApiConfig:
    """FastAPI configuration."""
    
    HOST: str = os.getenv("API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    BASE_URL: str = os.getenv("BASE_URL", f"http://{HOST}:{PORT}")


class LogConfig:
    """Logging configuration."""
    
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FILE: Path = LOGS_DIR / os.getenv("LOG_FILE", "app.log")
    ENABLE_CONSOLE: bool = os.getenv("ENABLE_CONSOLE_LOG", "true").lower() == "true"


class MongoDbConfig:
    """MongoDB configuration."""
    CONNECTION_STRING: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("MONGODB_DB", "face_attendance_db")
    COMPANY_ID: str = os.getenv("COMPANY_ID", "default_company")


class WebhookConfig:
    """Webhook configuration for attendance notifications."""
    USER_WEBHOOK_URL: str = os.getenv("USER_WEBHOOK_URL", "https://voice-cheking.bittechx.cloud/api/webhooks/user")


# Export all configs
__all__ = [
    "PROJECT_ROOT",
    "LOGS_DIR",
    "DATA_DIR",
    "CAPTURES_DIR",
    "EMBEDDINGS_DIR",
    "MODELS_DIR",
    "CameraConfig",
    "InsightFaceConfig",
    "RecognitionConfig",
    "QdrantConfig",
    "ApiConfig",
    "LogConfig",
    "AuthServiceConfig",
    "MongoDbConfig",
    "WebhookConfig",
]
