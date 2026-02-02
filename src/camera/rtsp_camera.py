import time
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.config import CameraConfig


class RTSPCamera:
    """
    RTSP Camera handler with Multi-threading for real-time streaming.
    
    This class uses a separate thread to constantly grab frames from the 
    RTSP stream, preventing buffer buildup and ensuring zero latency.
    """
    
    def __init__(
        self,
        rtsp_url: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        reconnect_delay: int = 5,
    ):
        self.rtsp_url = rtsp_url or CameraConfig.RTSP_URL
        self.width = width or CameraConfig.WIDTH
        self.height = height or CameraConfig.HEIGHT
        self.fps = fps or CameraConfig.FPS
        self.reconnect_delay = reconnect_delay
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        logger.info(f"RTSPCamera initialized in STREAM mode: {self._mask_url(self.rtsp_url)}")

    def _mask_url(self, url: str) -> str:
        if "@" in url:
            parts = url.split("@")
            return f"rtsp://***:***@{parts[-1]}"
        return url

    def connect(self) -> bool:
        """Connect to stream and start background thread."""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            # Optimization for RTSP
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                logger.error("Failed to open RTSP stream")
                return False

            self.is_connected = True
            
            # Start the background frame grabber thread
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                
            logger.success("Successfully connected and started background streaming thread.")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to RTSP: {e}")
            return False

    def _update(self):
        """Background thread: continuously grab frames from the stream."""
        while self.running:
            if self.is_connected and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    logger.warning("Stream connection lost in background thread.")
                    self.is_connected = False
            else:
                time.sleep(0.1)  # Wait for reconnection

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the ABSOLUTE LATEST frame from the background thread.
        This provides zero-latency performance.
        """
        if not self.is_connected:
            return False, None
            
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def reconnect(self) -> bool:
        """Reconnect logic."""
        logger.warning(f"Reconnecting in {self.reconnect_delay} seconds...")
        self.is_connected = False
        time.sleep(self.reconnect_delay)
        
        if self.cap:
            self.cap.release()
            
        return self.connect()

    def disconnect(self) -> None:
        """Stop thread and release camera."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.is_connected = False
        logger.info("Disconnected from RTSP stream")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        self.disconnect()

__all__ = ["RTSPCamera"]
