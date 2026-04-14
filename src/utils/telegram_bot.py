import platform
import socket
import requests
import cv2
import numpy as np
import threading
from loguru import logger
from src.config import TelegramConfig, CameraConfig, AuthServiceConfig
from src.utils.time_manager import time_mgr


# ---------------------------------------------------------------------------
# Device info block (dùng chung cho mọi report)
# ---------------------------------------------------------------------------

def _build_device_block() -> str:
    """
    Trả về chuỗi HTML mô tả thiết bị hiện tại,
    được đính kèm vào cuối mọi tin nhắn Telegram report.
    """
    # Host
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "unknown"

    os_name    = platform.system()
    os_ver     = platform.version()
    machine    = platform.machine()
    py_ver     = platform.python_version()

    # Camera — mask password
    rtsp_raw = getattr(CameraConfig, "RTSP_URL", "")
    try:
        if "@" in rtsp_raw and "://" in rtsp_raw:
            scheme, rest    = rtsp_raw.split("://", 1)
            userinfo, host_ = rest.rsplit("@", 1)
            user, _         = userinfo.split(":", 1) if ":" in userinfo else (userinfo, "")
            rtsp_display    = f"{scheme}://{user}:***@{host_}"
        else:
            rtsp_display = rtsp_raw or "N/A"
    except Exception:
        rtsp_display = rtsp_raw or "N/A"

    cam_name   = getattr(CameraConfig, "CAMERA_NAME", "Main Camera")
    resolution = f"{CameraConfig.WIDTH}x{CameraConfig.HEIGHT}"
    fps        = CameraConfig.FPS
    system_id  = AuthServiceConfig.SYSTEM_ID

    return (
        f"\n➖➖➖➖➖➖➖➖➖➖\n"
        f"📟 <b>DEVICE INFO</b>\n"
        f"🆔 <b>System ID:</b>\n"
        f"   <code>{system_id}</code>\n"
        f"\n"
        f"🖥️ <b>Host</b>\n"
        f"  ├ <b>Hostname:</b> <code>{hostname}</code>\n"
        f"  ├ <b>IP:</b>       <code>{ip}</code>\n"
        f"  ├ <b>OS:</b>       <code>{os_name}</code>\n"
        f"  ├ <b>Arch:</b>     <code>{machine}</code>\n"
        f"  └ <b>Python:</b>   <code>{py_ver}</code>\n"
        f"\n"
        f"📷 <b>Camera</b>\n"
        f"  ├ <b>Name:</b>       <code>{cam_name}</code>\n"
        f"  ├ <b>RTSP:</b>       <code>{rtsp_display}</code>\n"
        f"  ├ <b>Resolution:</b> <code>{resolution}</code>\n"
        f"  └ <b>FPS:</b>        <code>{fps}</code>"
    )


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _send_request(url, data=None, files=None):
    """Internal helper to send requests to Telegram API."""
    try:
        response = requests.post(url, data=data, files=files, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Telegram API error: {response.status_code} - {response.text}")
        return response
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_telegram_report(title, message, image=None, level="ERROR"):
    """
    Sends an automated report to Telegram.

    Tự động đính kèm thông tin devices (hostname, IP, OS, camera)
    vào cuối mỗi tin nhắn.

    Args:
        title (str): Error category (e.g., 'Sync Fail', 'Webhook Fail', 'Spoof Detected')
        message (str): Detailed error information
        image (Optional): OpenCV Frame (numpy array) or WebP bytes
        level (str): Log level emoji (default: ERROR)
    """
    if not TelegramConfig.ENABLED or not TelegramConfig.BOT_TOKEN or not TelegramConfig.CHAT_ID:
        return

    # Determine emoji based on level or title
    emoji = "🚨"
    if "RECOVERY" in title.upper() or "SUCCESS" in title.upper():
        emoji = "✅"
    elif "SPOOF" in title.upper():
        emoji = "🛡️"
    elif "SYNC" in title.upper():
        emoji = "🔄"
    elif "WEBHOOK" in title.upper():
        emoji = "🔗"

    timestamp, _ = time_mgr.get_formatted_time()

    # Build device info block (best-effort, never raises)
    try:
        device_block = _build_device_block()
    except Exception:
        device_block = ""

    caption = (
        f"{emoji} <b>{title.upper()} - REPORT</b>\n"
        f"📅 <b>Time:</b> <code>{timestamp}</code>\n"
        f"❌ <b>Details:</b>\n<i>{message}</i>"
        f"{device_block}"
    )

    def _send_text_only():
        token   = TelegramConfig.BOT_TOKEN
        chat_id = TelegramConfig.CHAT_ID
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        data    = {
            "chat_id":    chat_id,
            "text":       caption,
            "parse_mode": "HTML",
        }
        _send_request(url, data=data)

    def thread_task():
        token   = TelegramConfig.BOT_TOKEN
        chat_id = TelegramConfig.CHAT_ID

        # Debug log (masked token)
        token_preview = f"{token[:10]}...{token[-5:]}" if token else "None"
        logger.info(f"[Telegram] Sending to ChatID: {chat_id} using BotToken: {token_preview}")

        if image is not None:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"

            photo_data = None
            if isinstance(image, np.ndarray):
                success, encoded_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    photo_data = encoded_img.tobytes()
            else:
                photo_data = image  # Assume bytes

            if photo_data:
                # Telegram caption max = 1024 chars — truncate if needed
                safe_caption = caption[:1020] + "…" if len(caption) > 1024 else caption
                files = {"photo": ("report.jpg", photo_data, "image/jpeg")}
                data  = {
                    "chat_id":    chat_id,
                    "caption":    safe_caption,
                    "parse_mode": "HTML",
                }
                _send_request(url, data=data, files=files)
            else:
                _send_text_only()
        else:
            _send_text_only()

    # Run in background to not block the main process
    threading.Thread(target=thread_task, daemon=True).start()
