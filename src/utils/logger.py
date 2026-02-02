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
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=LogConfig.LEVEL,
            colorize=True,
        )
    
    # Add file logger with rotation
    logger.add(
        LogConfig.FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LogConfig.LEVEL,
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        encoding="utf-8",
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
