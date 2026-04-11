"""
Ping Service - Tự động gọi /api/v1/systems/ping mỗi 2 phút.

Service này chạy trong một background thread riêng biệt.
Nó sẽ hoạt động miễn là camera app còn đang chạy và stop_event chưa được set.

Payload gửi lên:
    {
        "system_id":   "848bbb0f-...",
        "ping_time":   "2026-04-07T10:30:00+07:00",
        "devices_info": {
            "camera": { "rtsp_url": "...", "resolution": "...", ... },
            "host":   { "os": "...", "hostname": "...", ... }
        }
    }
"""

import platform
import socket
import threading
import datetime
import requests
from loguru import logger

from src.config import AuthServiceConfig, CameraConfig, ReportSystemConfig


# ---------------------------------------------------------------------------
# Helper: thu thập thông tin thiết bị
# ---------------------------------------------------------------------------

def _build_devices_info() -> dict:
    """
    Trả về dict (map) mô tả các thiết bị đang hoạt động.
    Mỗi key là tên thiết bị, value là dict thông tin liên quan.
    """
    # --- Thông tin máy chủ (host) ---
    host_info: dict = {}
    try:
        host_info["hostname"] = socket.gethostname()
    except Exception:
        host_info["hostname"] = "unknown"

    try:
        host_info["ip"] = socket.gethostbyname(host_info.get("hostname", ""))
    except Exception:
        host_info["ip"] = "unknown"

    host_info["os"] = platform.system()
    host_info["os_version"] = platform.version()
    host_info["python_version"] = platform.python_version()
    host_info["machine"] = platform.machine()

    # --- Thông tin camera ---
    camera_info: dict = {
        "rtsp_url": _mask_rtsp_url(CameraConfig.RTSP_URL),
        "resolution": f"{CameraConfig.WIDTH}x{CameraConfig.HEIGHT}",
        "fps": CameraConfig.FPS,
        "flip_h": CameraConfig.FLIP_H,
        "flip_v": CameraConfig.FLIP_V,
        "name": getattr(CameraConfig, "CAMERA_NAME", "Main Camera"),
    }

    return {
        "host": host_info,
        "camera": camera_info,
    }


def _mask_rtsp_url(url: str) -> str:
    """
    Ẩn password trong RTSP URL trước khi gửi lên server.
    rtsp://user:pass@host → rtsp://user:***@host
    """
    try:
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            userinfo, hostpart = rest.rsplit("@", 1)
            if ":" in userinfo:
                user, _ = userinfo.split(":", 1)
                return f"{scheme}://{user}:***@{hostpart}"
    except Exception:
        pass
    return url


def _build_ping_time() -> str:
    """Trả về ISO 8601 timestamp với timezone +07:00."""
    tz = datetime.timezone(datetime.timedelta(hours=7))
    return datetime.datetime.now(tz).isoformat()


# ---------------------------------------------------------------------------
# PingService
# ---------------------------------------------------------------------------

class PingService:
    """
    Background service tự động ping Report System mỗi 2 phút.

    Cách dùng:
        service = PingService()
        service.start()     # Khởi động background thread
        # ... app chạy ...
        service.stop()      # Dừng sạch khi app tắt
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Khởi động background ping thread."""
        base = ReportSystemConfig.base_url()
        logger.info(
            f"PingService: Khởi động — kiểm tra cấu hình..."
            f" Report_System_Base_URL={repr(base)}"
        )

        if not ReportSystemConfig.is_enabled():
            logger.warning(
                "PingService: ❌ Report_System_Base_URL chưa được cấu hình trong .env. "
                "Ping service sẽ không chạy."
            )
            return

        logger.info(
            f"PingService: ✅ Sẽ ping {ReportSystemConfig.ping_url()} "
            f"mỗi {ReportSystemConfig.PING_INTERVAL_SECONDS}s."
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="PingServiceThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Dừng background ping thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("PingService: Đã dừng.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Vòng lặp chính: ping ngay lần đầu, rồi mỗi PING_INTERVAL_SECONDS."""
        self._ping()  # Ping ngay khi khởi động

        while not self._stop_event.is_set():
            # Chờ đúng khoảng thời gian nhưng vẫn phản hồi stop() ngay lập tức
            self._stop_event.wait(timeout=ReportSystemConfig.PING_INTERVAL_SECONDS)
            if self._stop_event.is_set():
                break
            self._ping()

    def _build_payload(self) -> dict:
        """Xây dựng JSON payload gửi lên server."""
        return {
            "system_id":    AuthServiceConfig.SYSTEM_ID,
            "system_name":  AuthServiceConfig.SYSTEM_NAME,
            "ping_time":    _build_ping_time(),
            "devices_info": _build_devices_info(),
        }

    def _ping(self):
        """Thực hiện một lần HTTP POST tới endpoint ping."""
        url = ReportSystemConfig.ping_url()
        payload = self._build_payload()

        logger.info(
            f"PingService: 📡 Đang gửi ping → {url} | "
            f"system_id={payload['system_id']} | ping_time={payload['ping_time']}"
        )

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code in (200, 201, 204):
                logger.info(
                    f"PingService: ✅ Ping thành công — [{response.status_code}]"
                )
            else:
                logger.warning(
                    f"PingService: ⚠️ Ping trả về status không mong đợi — "
                    f"[{response.status_code}] {response.text[:300]}"
                )

        except requests.exceptions.ConnectionError:
            logger.warning(
                f"PingService: ❌ Không thể kết nối tới {url}. Sẽ thử lại sau."
            )
        except requests.exceptions.Timeout:
            logger.warning(
                f"PingService: ⏱️ Timeout khi ping {url}. Sẽ thử lại sau."
            )
        except Exception as e:
            logger.error(f"PingService: Lỗi không xác định khi ping: {e}")


# Singleton instance — dùng chung trong toàn bộ app
ping_service = PingService()
