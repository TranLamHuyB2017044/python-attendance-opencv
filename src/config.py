"""
Configuration module for the attendance system.
Loads environment variables and provides centralized configuration.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from loguru import logger

# Project paths
if getattr(sys, 'frozen', False):
    # If running as a built .exe, PROJECT_ROOT is the folder where .exe is located
    PROJECT_ROOT = Path(os.path.dirname(sys.executable))
else:
    # If running in dev mode, PROJECT_ROOT is parent of src/
    PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables
if getattr(sys, 'frozen', False):
    # If running as a built .exe, look for .env in the same folder as the .exe
    env_path = PROJECT_ROOT / '.env'
    load_dotenv(dotenv_path=env_path, override=True)
    # Also fallback to default if not found
    if not env_path.exists():
        load_dotenv(override=True)
else:
    # If running in dev mode, always point to project root .env
    env_path = PROJECT_ROOT / '.env'
    load_dotenv(dotenv_path=env_path, override=True)

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
    
    # Detailed components for RTSP
    IP: str = ""
    PORT: int = 554
    USER: str = ""
    PASS: str = ""
    ROI = None
    
    # Priority 2: raw RTSP_URL from .env
    RTSP_URL: str = os.getenv("RTSP_URL", "")
    
    WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    FPS: int = max(1, int(os.getenv("CAMERA_FPS", "30")))  # Guard: tránh FPS=0 gây ZeroDivisionError
    FLIP_H: bool = os.getenv("CAMERA_FLIP_H", "false").lower() == "true"
    FLIP_V: bool = os.getenv("CAMERA_FLIP_V", "false").lower() == "true"
    CAMERA_NAME: str = os.getenv("CAMERA_NAME", "Main Camera")
    ROI_SIZE: Tuple[int, int] = (1280, 720) # Match default resolution, no cropping
    RTSP_PATH: str = "/ch1/main"  # Path + query (VD Dahua: /cam/realmonitor?channel=1&subtype=0)
    
    @classmethod
    def build_rtsp_url(cls, ip: str, port, user: str, pwd: str, path: str = "/ch1/main") -> str:
        """Ghép URL RTSP từ thành phần (path phải bắt đầu bằng /)."""
        path = (path or "/ch1/main").strip()
        if not path.startswith("/"):
            path = "/" + path
        return f"rtsp://{user}:{pwd}@{ip}:{int(port or 554)}{path}"

    @classmethod
    def rtsp_path_from_url(cls, url: str) -> str:
        """Lấy path+query từ URL RTSP đầy đủ."""
        if not url:
            return "/ch1/main"
        try:
            parsed = urlparse(url)
            path = parsed.path or "/"
            return f"{path}?{parsed.query}" if parsed.query else path
        except Exception:
            return "/ch1/main"

    @classmethod
    def parse_rtsp_url(cls, url: str) -> dict:
        """Tách ip/port/user/pass/path từ URL RTSP đầy đủ."""
        empty = {"ip": "", "port": 554, "user": "", "pass": "", "path": "/ch1/main"}
        if not url or not str(url).strip().lower().startswith("rtsp://"):
            return empty
        try:
            parsed = urlparse(url.strip())
            return {
                "ip": parsed.hostname or "",
                "port": parsed.port or 554,
                "user": unquote(parsed.username or ""),
                "pass": unquote(parsed.password or ""),
                "path": cls.rtsp_path_from_url(url),
            }
        except Exception:
            return empty

    @classmethod
    def resolve_rtsp_url_for_company(cls, mongo_db, company_id: str) -> str:
        """
        Đọc cấu hình camera theo company_id (trùng key username trong MongoDB settings).
        Dùng chung cho service, GUI và hot-reload.
        """
        cid = (company_id or "").strip()

        # Ưu tiên field rời (ip/port/user/pass/path) — cập nhật từ xa chỉ cần đổi camera_ip
        ip = mongo_db.get_setting("camera_ip", None, username=cid)
        port = mongo_db.get_setting("camera_port", None, username=cid)
        user = mongo_db.get_setting("camera_user", None, username=cid)
        pwd = mongo_db.get_setting("camera_pass", None, username=cid)
        path = mongo_db.get_setting("camera_rtsp_path", None, username=cid)

        if ip and user and pwd:
            if not path:
                path = "/ch1/main"
            url = cls.build_rtsp_url(ip, port, user, pwd, path)
            logger.info(
                f"CameraConfig.resolve: company_id={cid} | "
                f"source=MongoDB fields (remote OK) | user={user} | ip={ip} | "
                f"url={cls._mask_rtsp(url)}"
            )
            return url

        full_url = (mongo_db.get_setting("camera_rtsp_url", None, username=cid) or "").strip()
        if full_url.lower().startswith("rtsp://"):
            logger.info(
                f"CameraConfig.resolve: company_id={cid} | "
                f"source=MongoDB camera_rtsp_url (fallback) | url={cls._mask_rtsp(full_url)}"
            )
            return full_url

        env_url = os.getenv("RTSP_URL", "").strip()
        if env_url:
            logger.info(
                f"CameraConfig.resolve: company_id={cid} | "
                f"source=.env (MongoDB thiếu ip/user/pass) | url={cls._mask_rtsp(env_url)}"
            )
            return env_url

        logger.warning(f"CameraConfig.resolve: company_id={cid} | không có cấu hình camera")
        return ""

    @staticmethod
    def _mask_rtsp(url: str) -> str:
        if "@" in url:
            return f"rtsp://***:***@{url.split('@', 1)[-1]}"
        return url

    @classmethod
    def load_from_mongodb(cls, mongo_db):
        """
        Fetch latest camera settings from MongoDB and update the static config.
        This allows the app and background service to be configured via the Management UI.
        """
        try:
            from src.config import MongoDbConfig, RecognitionConfig
            cid = MongoDbConfig.COMPANY_ID
            logger.info(f"CameraConfig.load_from_mongodb: COMPANY_ID từ .env = {cid}")

            ip = mongo_db.get_setting("camera_ip", None, username=cid)
            port = mongo_db.get_setting("camera_port", None, username=cid)
            user = mongo_db.get_setting("camera_user", None, username=cid)
            pwd = mongo_db.get_setting("camera_pass", None, username=cid)
            path = mongo_db.get_setting("camera_rtsp_path", None, username=cid)

            if ip and user and pwd:
                if not path:
                    path = "/ch1/main"
                cls.IP = ip
                cls.PORT = int(port or "554")
                cls.USER = user
                cls.PASS = pwd
                cls.RTSP_PATH = path
                cls.RTSP_URL = cls.build_rtsp_url(ip, port, user, pwd, path)
                logger.info(
                    f"CameraConfig: MongoDB fields cho company_id={cid} | "
                    f"ip={ip} | url={cls._mask_rtsp(cls.RTSP_URL)}"
                )
            else:
                full_url = (mongo_db.get_setting("camera_rtsp_url", None, username=cid) or "").strip()
                if full_url.lower().startswith("rtsp://"):
                    creds = cls.parse_rtsp_url(full_url)
                    cls.IP = creds["ip"]
                    cls.PORT = int(creds["port"] or 554)
                    cls.USER = creds["user"]
                    cls.PASS = creds["pass"]
                    cls.RTSP_PATH = creds["path"]
                    cls.RTSP_URL = full_url
                    logger.info(
                        f"CameraConfig: fallback camera_rtsp_url company_id={cid} | "
                        f"url={cls._mask_rtsp(cls.RTSP_URL)}"
                    )
                else:
                    cls.RTSP_URL = os.getenv("RTSP_URL", "").strip()
                    cls.RTSP_PATH = cls.rtsp_path_from_url(cls.RTSP_URL) if cls.RTSP_URL else "/ch1/main"
                    if cls.RTSP_URL:
                        creds = cls.parse_rtsp_url(cls.RTSP_URL)
                        cls.IP = creds["ip"]
                        cls.PORT = int(creds["port"] or 554)
                        cls.USER = creds["user"]
                        cls.PASS = creds["pass"]
                        logger.info(
                            f"CameraConfig: MongoDB trống, bootstrap từ .env | "
                            f"url={cls._mask_rtsp(cls.RTSP_URL)}"
                        )
                    else:
                        logger.warning(
                            f"CameraConfig: Không có camera config cho company_id={cid}"
                        )
            
            # 4. Update Recognition & Cooldown Settings
            cooldown_sec = mongo_db.get_setting("detection_cooldown", str(RecognitionConfig.COOLDOWN_SECONDS), username=cid)
            RecognitionConfig.COOLDOWN_SECONDS = int(cooldown_sec)
            
            # Anti-spoofing config
            anti_spoofing = mongo_db.get_setting("anti_spoofing_enabled", str(RecognitionConfig.ANTI_SPOOFING_ENABLED), username=cid)
            RecognitionConfig.ANTI_SPOOFING_ENABLED = str(anti_spoofing).lower() == "true"

            # Telegram config (cho phép bật tắt từ xa)
            telegram_enabled = mongo_db.get_setting("enable_telegram_notif", str(TelegramConfig.ENABLED), username=cid)
            TelegramConfig.ENABLED = str(telegram_enabled).lower() == "true"


            # Recognition threshold — điều chỉnh từ xa để tăng/giảm độ chính xác
            # Key MongoDB: "recognition_threshold", VD: "0.65"
            threshold_str = mongo_db.get_setting("recognition_threshold", str(RecognitionConfig.THRESHOLD), username=cid)
            try:
                RecognitionConfig.THRESHOLD = float(threshold_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid recognition_threshold from MongoDB: '{threshold_str}', keeping current: {RecognitionConfig.THRESHOLD}")

            # Gather frames — số frame tích lũy trước khi nhận diện
            # Key MongoDB: "gather_frames", VD: "1" (nhanh) hoặc "5" (chính xác hơn)
            gather_str = mongo_db.get_setting("gather_frames", str(RecognitionConfig.GATHER_FRAMES), username=cid)
            try:
                RecognitionConfig.GATHER_FRAMES = max(1, int(gather_str))  # Tối thiểu 1
            except (ValueError, TypeError):
                logger.warning(f"Invalid gather_frames from MongoDB: '{gather_str}', keeping current: {RecognitionConfig.GATHER_FRAMES}")

            # ROI Configuration
            roi_str = mongo_db.get_setting("camera_roi", "", username=cid)
            try:
                if roi_str and len(roi_str.split(',')) == 4:
                    cls.ROI = tuple(map(int, roi_str.split(',')))
                else:
                    cls.ROI = None
            except:
                cls.ROI = None

            logger.info(
                f"CameraConfig: Updated settings from MongoDB -> company_id={cid} | "
                f"{cls.IP}:{cls.PORT} path={getattr(cls, 'RTSP_PATH', '/ch1/main')} | "
                f"rtsp={cls._mask_rtsp(cls.RTSP_URL)} | "
                f"Cooldown: {cooldown_sec}s | Anti-Spoof: {RecognitionConfig.ANTI_SPOOFING_ENABLED} | "
                f"Threshold: {RecognitionConfig.THRESHOLD} | GatherFrames: {RecognitionConfig.GATHER_FRAMES} | "
                f"ROI: {cls.ROI}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load camera settings from MongoDB: {e}")
            return False

    @classmethod
    def validate(cls) -> bool:
        """Validate camera configuration."""
        if not cls.RTSP_URL or cls.RTSP_URL == "":
            logger.warning("RTSP_URL is not set and no fallback is permitted!")
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
    
    THRESHOLD: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.6"))
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))
    TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    ANTI_SPOOFING_ENABLED: bool = os.getenv("ANTI_SPOOFING_ENABLED", "true").lower() == "true"
    COOLDOWN_SECONDS: int = int(os.getenv("DETECTION_COOLDOWN", "3600")) # Default 1 hour
    MAX_FACES: int = int(os.getenv("MAX_FACES", "100"))
    CAPTURE_MAX_WIDTH: int = int(os.getenv("CAPTURE_MAX_WIDTH", "1280"))
    MIN_FACE_SIZE: int = int(os.getenv("MIN_FACE_SIZE", "60"))  # Giảm từ 80 xuống 60 để detect mặt nhỏ hơn
    # Số frame thu thập trước khi kết luận nhận diện (Best-of-N voting)
    # Tăng lên nhưng chậm hơn (khuyên dùng 3-7), giảm xuống nhưng nhanh hơn
    GATHER_FRAMES: int = int(os.getenv("GATHER_FRAMES", "3"))

    # Ngưỡng chất lượng frame trước gather/recognize
    MIN_BLUR_VARIANCE: float = float(os.getenv("MIN_BLUR_VARIANCE", "50"))
    # Chỉ siết xoay ngang (yaw); pitch giữ 30° cho người cao/thấp trước camera
    MAX_YAW_DEG: float = float(os.getenv("MAX_YAW_DEG", "20"))
    MAX_PITCH_DEG: float = float(os.getenv("MAX_PITCH_DEG", "30"))
    
    # Cấu hình mới cho detection đa góc
    DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))  # Giảm confidence threshold
    ENABLE_MULTI_ANGLE: bool = os.getenv("ENABLE_MULTI_ANGLE", "true").lower() == "true"
    FACE_DETECTION_INTERVAL: int = int(os.getenv("FACE_DETECTION_INTERVAL", "1"))  # Detect mỗi frame


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
    SYSTEM_NAME: str = os.getenv("SYSTEM_NAME", "FACE AI CHECKING")
    SYSTEM_REGISTER: str = os.getenv("SYSTEM_REGISTER", "")
    CURRENT_USER_ID: int = 1  # Global session tracking for reporting


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
    DATABASE_NAME: str = os.getenv("MONGODB_DB", "attendance_system")
    COMPANY_ID: str = os.getenv("COMPANY_ID", "default_company")


class EmailConfig:
    """SMTP Email configuration for password reset."""
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "") # App Password for Gmail


class TelegramConfig:
    """Telegram Bot configuration for error reporting."""
    _raw_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _raw_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    BOT_TOKEN: str = _raw_token.strip()
    CHAT_ID: str = _raw_chat_id.strip()
    ENABLED: bool = os.getenv("ENABLE_TELEGRAM_NOTIF", "false").lower() == "true"


class SpecialUserConfig:
    """Cấu hình đặc biệt cho nhân viên cụ thể"""
    # Định nghĩa các user đặc biệt: {user_id: {"voice_text": "...", "cooldown_hours": ...}}
    SPECIAL_USERS = {
        "7980041": {
            "voice_text": "Chào anh Thanh",
            "cooldown_hours": 1
        }
    }
    
    @classmethod
    def is_special_user(cls, user_id):
        return str(user_id) in cls.SPECIAL_USERS
    
    @classmethod
    def get_voice_text(cls, user_id):
        return cls.SPECIAL_USERS.get(str(user_id), {}).get("voice_text")
    
    @classmethod
    def get_cooldown_seconds(cls, user_id):
        hours = cls.SPECIAL_USERS.get(str(user_id), {}).get("cooldown_hours", 0)
        return hours * 3600


class WebhookConfig:
    """Webhook configuration for attendance notifications."""
    USER_WEBHOOK_URL: str = os.getenv("USER_WEBHOOK_URL", "https://voice-cheking.bittechx.cloud/api/webhooks/user")


class ReportSystemConfig:
    """Report System configuration for periodic ping."""
    PING_INTERVAL_SECONDS: int = 120  # Ping mỗi 2 phút
    PING_ENDPOINT: str = "/api/v1/systems/ping"
    REPORT_LOG_ENDPOINT: str = "/api/v1/report-log"

    @classmethod
    def base_url(cls) -> str:
        """Returns the base URL from env, evaluated at call time (not import time)."""
        return os.getenv("Report_System_Base_URL", "").rstrip("/")

    @classmethod
    def ping_url(cls) -> str:
        """Returns the full ping URL."""
        return f"{cls.base_url()}{cls.PING_ENDPOINT}"

    @classmethod
    def report_log_url(cls) -> str:
        """Returns the full report log URL."""
        return f"{cls.base_url()}{cls.REPORT_LOG_ENDPOINT}"

    @classmethod
    def is_enabled(cls) -> bool:
        """Returns True if the Report System URL is configured."""
        return bool(cls.base_url())

    # Backward compat: keep BASE_URL as property-like fallback
    BASE_URL: str = ""  # deprecated, dùng base_url() thay thế


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
    "TelegramConfig",
    "ReportSystemConfig",
    "SpecialUserConfig",
]
