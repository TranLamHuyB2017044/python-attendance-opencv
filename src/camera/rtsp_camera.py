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
        reconnect_delay: int = 3,
    ):
        self.rtsp_url = rtsp_url or CameraConfig.RTSP_URL
        
        # Convert to int if it's a numeric string (webcam index)
        try:
            self.camera_source = int(self.rtsp_url)
            self.is_webcam = True
        except (ValueError, TypeError):
            self.camera_source = self.rtsp_url
            self.is_webcam = False
        
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
        
        if self.is_webcam:
            logger.info(f"RTSPCamera initialized with WEBCAM mode: index {self.camera_source}")
        else:
            logger.info(f"RTSPCamera initialized in STREAM mode: {self._mask_url(str(self.camera_source))}")

    def _mask_url(self, url: str) -> str:
        if "@" in url:
            parts = url.split("@")
            return f"rtsp://***:***@{parts[-1]}"
        return url

    def _connect_with_timeout(self, timeout_seconds=10):
        """Try to connect with timeout to prevent blocking."""
        result = {"success": False, "cap": None}
        
        def _try_connect():
            try:
                cap = cv2.VideoCapture(self.camera_source)
                if self.is_webcam:
                    # Webcam settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    # RTSP settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                
                if cap.isOpened():
                    result["cap"] = cap
                    result["success"] = True
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                logger.error(f"Exception during connection: {e}")
        
        # Run connection in separate thread with timeout
        connect_thread = threading.Thread(target=_try_connect, daemon=True)
        connect_thread.start()
        connect_thread.join(timeout=timeout_seconds)
        
        if connect_thread.is_alive():
            logger.error(f"Connection timeout after {timeout_seconds} seconds")
            return None
        
        return result["cap"] if result["success"] else None

    def connect(self) -> bool:
        """Connect to stream and start background thread."""
        try:
            logger.info(f"Attempting to connect to {'webcam' if self.is_webcam else 'RTSP camera'}...")
            
            # Use timeout connection (reduced from 10s to 5s to prevent long hangs)
            self.cap = self._connect_with_timeout(timeout_seconds=5)
            
            if self.cap is None or not self.cap.isOpened():
                logger.error(f"Failed to open {'webcam' if self.is_webcam else 'RTSP stream'}")
                return False

            self.is_connected = True
            
            # Start the background frame grabber thread
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                
            logger.success(f"Successfully connected to {'webcam' if self.is_webcam else 'RTSP stream'} and started background thread.")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {'webcam' if self.is_webcam else 'RTSP'}: {e}")
            return False

    def _update(self):
        """Background thread: continuously grab frames from the stream."""
        while self.running:
            if self.is_connected and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    # Apply Flip if configured
                    if CameraConfig.FLIP_H and CameraConfig.FLIP_V:
                        frame = cv2.flip(frame, -1) # Both
                    elif CameraConfig.FLIP_H:
                        frame = cv2.flip(frame, 1)  # Horizontal
                    elif CameraConfig.FLIP_V:
                        frame = cv2.flip(frame, 0)  # Vertical
                        
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
                frame = self.frame.copy()
                
                # No cropping - keep original camera size
                # h, w = frame.shape[:2]
                # roi_w, roi_h = CameraConfig.ROI_SIZE
                
                # if roi_w < w or roi_h < h:
                #     x1 = max(0, (w - roi_w) // 2)
                #     y1 = max(0, (h - roi_h) // 2)
                #     frame = frame[y1:y1+roi_h, x1:x1+roi_w]
                
                return True, frame
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
