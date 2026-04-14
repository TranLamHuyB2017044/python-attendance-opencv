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
        enable_notifications: bool = True,
    ):
        self.rtsp_url = rtsp_url or CameraConfig.RTSP_URL
        self.enable_notifications = enable_notifications
        logger.info(f"RTSPCamera initialized with URL: {self.rtsp_url}")
        
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
        self.ping_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        self.current_ping = "N/A"
        self.last_frame_time = 0
        self._notified_error = False # Track if we've already sent an Error notification
        
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
                if self.is_webcam:
                    import os
                    if os.name == 'nt':
                        cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(self.camera_source)
                    # Webcam settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    import os
                    # =============================================================
                    # MINIMUM LATENCY FFMPEG FLAGS (Tối ưu tối đa latency RTSP)
                    # =============================================================
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join([
                        "rtsp_transport;tcp",        # Force TCP (tránh mất gói UDP)
                        "fflags;nobuffer",           # Tắt FFmpeg demuxer buffer
                        "flags;low_delay",           # Low-latency decode mode
                        "avioflags;direct",          # I/O trực tiếp, không qua buffer OS
                        "probesize;32",              # Tối thiểu probe size (default=5MB!)
                        "analyzeduration;0",         # Không phân tích stream trước khi play
                        "reorder_queue_size;0",      # Không reorder gói (+50-100ms nếu bật)
                        "max_delay;50000",           # Jitter buffer 50ms — ổn định hơn 0 với camera IP
                        "stimeout;3000000",          # Socket timeout 3s (tránh treo khi mạng lag)
                        "framedrop",                 # Bỏ frame cũ nếu decode không kịp
                    ])

                    cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                    # Chỉ giữ 1 frame trong bộ đệm OpenCV (FFmpeg vẫn có buffer riêng)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
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
                if not self._notified_error and self.enable_notifications:
                    logger.error(f"Failed to open {'webcam' if self.is_webcam else 'RTSP stream'}")
                    self._notified_error = True # Mark that we've notified the error
                else:
                    # Subsequent failures log as warning to avoid Telegram spam
                    logger.warning(f"Still cannot connect to {'webcam' if self.is_webcam else 'RTSP stream'}...")
                return False

            # If we were in error state, notify recovery (if enabled)
            if self._notified_error and self.enable_notifications:
                try:
                    from src.utils.telegram_bot import send_telegram_report
                    send_telegram_report("CAMERA RECOVERY", f"Camera {'webcam' if self.is_webcam else 'RTSP'} connection has been restored.")
                except Exception:
                    pass
                logger.success(f"Camera {'webcam' if self.is_webcam else 'RTSP'} connection RECOVERED!")
                self._notified_error = False
            elif self._notified_error:
                # Still reset state even if notifications disabled
                self._notified_error = False

            self.is_connected = True
            
            # Start the background frame grabber thread
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                
                if not self.is_webcam:
                    self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
                    self.ping_thread.start()
                
            logger.success(f"Successfully connected to {'webcam' if self.is_webcam else 'RTSP stream'} and started background thread.")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {'webcam' if self.is_webcam else 'RTSP'}: {e}")
            return False

    def _ping_loop(self):
        """Background thread to check actual latency to the camera IP."""
        import socket
        try:
            url = str(self.camera_source)
            if "rtsp://" in url:
                host_part = url.split("rtsp://")[1].split("/")[0]
                if "@" in host_part:
                    host_part = host_part.split("@")[1]
                ip = host_part.split(":")[0]
                port = int(host_part.split(":")[1]) if ":" in host_part else 554
                
                while self.running:
                    try:
                        s_time = time.time()
                        with socket.create_connection((ip, port), timeout=2.0):
                            pass
                        e_time = time.time()
                        self.current_ping = f"{int((e_time - s_time) * 1000)}ms"
                    except Exception:
                        self.current_ping = "Timeout"
                    time.sleep(2)
        except Exception:
            self.current_ping = "Err"

    def _update(self):
        """
        Background thread: grab frames from RTSP with minimum latency.

        KEY INSIGHT về RTSP:
          cap.read() đối với RTSP tự block chờ frame mới từ mạng.
          Nếu chúng ta sleep() thêm, frame mới tới trong lúcng sleep
          và nằm trong buffer → lần read() tiếp theo lấy frame CŨ đó!
          → Sleep thêm = tự cộng thêm latency mà chúng ta vừa fix!

        Giải pháp:
          1. RTSP: Dùng grab() để xả sạch buffer, chỉ decode frame mới nhất.
          2. Webcam: Giữ adaptive sleep như cũ (cảm biến local, không có network buffer).
        """
        if self.is_webcam:
            # --- WEBCAM: giữ throttle cũ để tránh read nhanh hơn cảm biến ---
            # Nếu cfg FPS <= 0 -> chạy thả ga tẹt ga tự do với tốc độ tối đa của camera
            target_interval = 1.0 / self.fps if self.fps > 0 else 0.0
            while self.running:
                frame_start = time.time()
                if self.is_connected and self.cap is not None:
                    ret, frame = self.cap.read()
                    if ret:
                        if CameraConfig.FLIP_H and CameraConfig.FLIP_V: frame = cv2.flip(frame, -1)
                        elif CameraConfig.FLIP_H: frame = cv2.flip(frame, 1)
                        elif CameraConfig.FLIP_V: frame = cv2.flip(frame, 0)
                        with self.lock:
                            self.frame = frame
                    else:
                        logger.warning("Webcam disconnected.")
                        self.is_connected = False
                else:
                    time.sleep(0.1)
                    continue
                elapsed = time.time() - frame_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        else:
            # --- RTSP: MINIMUM LATENCY MODE ---
            #
            # THIẾT KẾ ĐÚNG CHO RTSP (Network Stream):
            #   Background thread liên tục đọc và CẬP NHẬT self.frame với frame mới nhất.
            #   Main thread gọi read_frame() → luôn nhận frame MỚI NHẤT đã được lưu.
            #
            # TẠI SAO KHÔNG DÙNG grab()-drain():
            #   Với RTSP, grab() CŨNG BLOCK chờ gói tin mới từ mạng — không có
            #   "buffer sẵn" để drain như video file. Dùng drain = gọi grab() nhiều
            #   lần liên tiếp = tăng latency thay vì giảm!
            #
            # GIẢI PHÁP: cap.read() trong thread riêng, luôn chạy hết tốc độ.
            #   Thread này chạy đúng tốc độ camera (15/25/30fps) do cap.read() tự block.
            while self.running:
                if not self.is_connected or self.cap is None:
                    time.sleep(0.1)
                    continue

                try:
                    ret, frame = self.cap.read()  # Block chờ frame mới từ mạng

                    if ret and frame is not None:
                        if CameraConfig.FLIP_H and CameraConfig.FLIP_V: frame = cv2.flip(frame, -1)
                        elif CameraConfig.FLIP_H: frame = cv2.flip(frame, 1)
                        elif CameraConfig.FLIP_V: frame = cv2.flip(frame, 0)
                        with self.lock:
                            self.frame = frame
                    else:
                        logger.warning("Stream connection lost in background thread.")
                        self.is_connected = False
                except Exception as e:
                    logger.error(f"[Camera] _update error: {e}")
                    self.is_connected = False

    def read_frame(self, wait_first_frame: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the ABSOLUTE LATEST frame from the background thread.
        This provides zero-latency performance.

        Args:
            wait_first_frame: Neu True, cho toi da 2s de background thread capture
                              frame dau tien (thay vi tra None ngay lap tuc).
        """
        if not self.is_connected:
            return False, None

        # Cho frame dau tien neu can (thread moi chay, cap.read() chua tra lai)
        if wait_first_frame and self.frame is None:
            deadline = time.time() + 2.0
            while self.frame is None and time.time() < deadline and self.is_connected:
                time.sleep(0.01)

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
