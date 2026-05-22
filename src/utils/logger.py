"""
Logging utility module.
Configures loguru for both console and file logging.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import LogConfig


def setup_logger() -> None:
    """
    Configure loguru logger with console and file handlers.
    
    This function:
    - Removes default logger
    - Adds console logger (if enabled)
    - Adds file logger with rotation
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger if enabled and available (sys.stdout is None in Windowed mode)
    if LogConfig.ENABLE_CONSOLE and sys.stdout is not None:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=LogConfig.LEVEL,
            colorize=True,
            catch=True, # Prevent app crash if console logging fails
        )
    
    # Add file logger with rotation
    logger.add(
        LogConfig.FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LogConfig.LEVEL,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        catch=True, # Prevent app crash if file logging fails
    )

    # Add MongoDB sink for WARNING and ERROR levels
    def mongodb_sink(message):
        record = message.record
        try:
            from src.attendance.mongodb_mgr import mongo_db
            from src.utils.time_manager import time_mgr
            vn_now = time_mgr.get_accurate_time()
            doc = {
                "timestamp": vn_now,
                "time_str": vn_now.strftime("%Y-%m-%d %H:%M:%S"),
                "level": record["level"].name,
                "message": record["message"],
                "source": f"{record['name']}:{record['function']}:{record['line']}"
            }
            mongo_db.db.system_logs.insert_one(doc)
        except Exception:
            pass

    logger.add(
        mongodb_sink,
        level="SUCCESS", # SUCCESS = 25, captures SUCCESS, WARNING, ERROR, CRITICAL
        enqueue=True, # Run in background thread automatically
        catch=True
    )

    # Add Telegram sink for ERROR and CRITICAL levels with Anti-Spam
    last_telegram_time = 0
    def telegram_sink(message):
        nonlocal last_telegram_time
        import time
        current_time = time.time()
        # Chỉ cho phép gửi Telegram mỗi 10 giây một lần để tránh Loop/Spam
        if current_time - last_telegram_time < 10:
            return
            
        from src.utils.time_manager import time_mgr
        # from src.services.report_service import report_service  <-- REMOVE THIS TO AVOID CIRCULAR IMPORT / LOOP
        from src.utils.telegram_bot import send_telegram_report
        record = message.record
        title = f"System {record['level'].name}"
        error_msg = f"{record['name']}:{record['function']}:{record['line']} - {record['message']}"
        
        last_telegram_time = current_time
        send_telegram_report(title, error_msg)

    logger.add(
        telegram_sink,
        level="ERROR",
        enqueue=True,
        catch=True
    )
    
    logger.info("Logger initialized successfully")


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        Logger: Configured loguru logger
    """
    return logger


__all__ = ["setup_logger", "get_logger"]
