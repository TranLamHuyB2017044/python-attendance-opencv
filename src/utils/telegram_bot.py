import requests
import cv2
import numpy as np
import threading
from loguru import logger
from src.config import TelegramConfig
from src.utils.time_manager import time_mgr

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

def send_telegram_report(title, message, image=None, level="ERROR"):
    """
    Sends an automated report to Telegram.
    
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
    if "SPOOF" in title.upper():
        emoji = "🛡️"
    elif "SYNC" in title.upper():
        emoji = "🔄"
    elif "WEBHOOK" in title.upper():
        emoji = "🔗"
    
    timestamp, _ = time_mgr.get_formatted_time()
    
    caption = (
        f"{emoji} <b>{title.upper()} - REPORT</b>\n"
        f"📅 <b>Time:</b> <code>{timestamp}</code>\n"
        f"❌ <b>Details:</b>\n<i>{message}</i>"
    )

    def thread_task():
        token = TelegramConfig.BOT_TOKEN
        chat_id = TelegramConfig.CHAT_ID
        
        # Debug log to verify loaded config (masked)
        token_preview = f"{token[:10]}...{token[-5:]}" if token else "None"
        logger.info(f"[Telegram] Sending to ChatID: {chat_id} using BotToken: {token_preview}")
        
        if image is not None:
            # Handle image sending
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            
            photo_data = None
            if isinstance(image, np.ndarray):
                # Convert OpenCV frame to JPEG/WebP bytes
                success, encoded_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    photo_data = encoded_img.tobytes()
            else:
                photo_data = image # Assume bytes

            if photo_data:
                files = {'photo': ('report.jpg', photo_data, 'image/jpeg')}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }
                _send_request(url, data=data, files=files)
            else:
                # Fallback to message if image conversion fails
                _send_text_only()
        else:
            _send_text_only()

    def _send_text_only():
        token = TelegramConfig.BOT_TOKEN
        chat_id = TelegramConfig.CHAT_ID
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': caption,
            'parse_mode': 'HTML'
        }
        _send_request(url, data=data)

    # Run in background to not block the main process
    threading.Thread(target=thread_task, daemon=True).start()
