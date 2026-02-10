import ctypes
import sys
import os
from loguru import logger

def show_error_message(title, message):
    """Shows a Windows Message Box with an Error icon. (Blocking)"""
    logger.error(f"FATAL ERROR: [{title}] {message}")
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10) # 0x10 is MB_ICONERROR
    else:
        print(f"ERROR: [{title}] {message}")

def show_info_message(title, message):
    """Shows a Windows Message Box with an Info icon. (Blocking)"""
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40) # 0x40 is MB_ICONINFORMATION
    else:
        print(f"INFO: [{title}] {message}")

def send_notification(title, message):
    """Sends a non-blocking toast notification (Windows Notification)."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name='Attendance AI System',
            app_icon=None, # You can provide path to .ico if you have one
            timeout=5, # Seconds
        )
        logger.info(f"Notification sent: {title} - {message}")
    except Exception as e:
        logger.warning(f"Could not send notification: {e}")
        # Fallback to simple print/log
        print(f"NOTIFY: [{title}] {message}")
