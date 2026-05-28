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
    
    # Add console logger if enabled
    if LogConfig.ENABLE_CONSOLE:
        try:
            # Try to use sys.stdout, if not available (frozen app), use sys.__stdout__
            stream = sys.stdout if sys.stdout is not None else sys.__stdout__
            logger.add(
                stream,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=LogConfig.LEVEL,
                colorize=True,
                catch=True, # Prevent app crash if console logging fails
            )
        except Exception as e:
            print(f"Warning: Could not setup console logger: {e}")
    
    # Add file logger with rotation
    try:
        # Ensure logs directory exists
        LogConfig.FILE.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            LogConfig.FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=LogConfig.LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            catch=True, # Prevent app crash if file logging fails
            enqueue=True, # Use queue for thread-safe logging
        )
        logger.info(f"File logger initialized, writing to: {LogConfig.FILE}")
    except Exception as e:
        print(f"Warning: Could not setup file logger: {e}")

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
    sent_errors = set()  # Track các lỗi đã gửi để không lặp lại
    
    def telegram_sink(message):
        nonlocal last_telegram_time, sent_errors
        import time
        current_time = time.time()
        
        record = message.record
        error_key = f"{record['name']}:{record['function']}:{record['line']}:{record['message'][:100]}"
        
        # Nếu lỗi này đã gửi rồi, bỏ qua
        if error_key in sent_errors:
            return
            
        # Chỉ cho phép gửi Telegram mỗi 30 giây một lần để tránh Loop/Spam
        if current_time - last_telegram_time < 30:
            return
            
        from src.utils.telegram_bot import send_telegram_report
        title = f"System {record['level'].name}"
        error_msg = f"{record['name']}:{record['function']}:{record['line']} - {record['message']}"
        
        last_telegram_time = current_time
        sent_errors.add(error_key)
        
        # Giới hạn số lỗi lưu trong bộ nhớ (tối đa 100 lỗi) để tránh leak memory
        if len(sent_errors) > 100:
            sent_errors.clear()
            
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
