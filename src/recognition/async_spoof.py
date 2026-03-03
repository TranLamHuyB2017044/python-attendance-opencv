"""
Anti-Spoofing Checker (Sync-on-Verdict Mode)
=============================================
Cơ chế: Không async pre-submit nữa.
Thay vào đó, chỉ chạy 1 lần ĐỒNG BỘ khi GATHERING xong với timeout 150ms.

Tại sao đổi cơ chế?
- Async submit: pool có thể chưa kịp xử lý trước get_verdict() → race condition
- Model loading race: model chưa load xong → safe default = REAL → phone ảnh qua
- Sync-on-verdict: chạy ngay khi quyết định, có kết quả ngay, chính xác hơn

Cách dùng trong tracker.py:
    # Sau khi GATHERING xong, trước khi RECOGNIZED:
    is_real, score = spoof_checker.check_sync(frame, face)
    if not is_real:
        → SPOOF_DETECTED
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Tuple
import numpy as np
import cv2
from loguru import logger


class AsyncSpoofChecker:  # Giữ tên cũ để không đổi import trong tracker.py
    """
    Anti-spoofing checker — chạy ĐỒNG BỘ tại thời điểm kết luận.

    Gọi check_sync(frame, face) sau khi GATHERING xong → kết quả ngay.
    Có timeout 150ms để không block lâu nếu model chậm.

    2 lớp kiểm tra:
      Lớp 1: Laplacian blur score (~1ms) — ảnh in mờ, màn hình chất lượng thấp
      Lớp 2: MiniFASNetV2 ML model (~20-40ms) — ảnh HD, video, mặt nạ
    """

    def __init__(self, model_dir: str = "models/models/anti_spoof"):
        self._model_dir  = model_dir
        self._model      = None
        self._model_loaded = False
        self._load_lock  = threading.Lock()
        self._pool       = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spoof_sync")

        # Load model trong background để không block startup
        threading.Thread(target=self._load_model, daemon=True, name="spoof_loader").start()

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _load_model(self):
        """Load MiniFASNet model một lần. Gọi từ background thread."""
        try:
            from src.config import MODELS_DIR
            import os
            if self._model_dir is None:
                self._model_dir = str(MODELS_DIR / "anti_spoof")
            
            if not os.path.exists(self._model_dir):
                logger.warning(f"[Spoof] Model dir not found: {self._model_dir} → DISABLED")
                return

            from src.recognition.antispoofing import AntiSpoofing
            model = AntiSpoofing(model_dir=self._model_dir)
            with self._load_lock:
                self._model        = model
                self._model_loaded = True
            logger.success("[Spoof] MiniFASNetV2 loaded — anti-spoofing ACTIVE")
        except Exception as e:
            logger.error(f"[Spoof] Failed to load model: {e}")

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Chờ model load xong (chỉ gọi lúc startup nếu cần block)."""
        t0 = time.time()
        while not self._model_loaded and (time.time() - t0) < timeout:
            time.sleep(0.1)
        return self._model_loaded

    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._model is not None

    # ─── Public API ───────────────────────────────────────────────────────────

    def check_sync(self, frame: np.ndarray, face,
                   timeout: float = 0.15) -> Tuple[bool, float]:
        """
        Chạy spoof check ĐỒNG BỘ với timeout.
        Gọi 1 lần sau khi GATHERING xong — thay thế submit() + get_verdict().

        Returns:
            (is_real, score)
            - is_real=True  → người thật, tiếp tục chấm công
            - is_real=False → giả mạo, từ chối
            - is_real=True  nếu model chưa sẵn hoặc timeout (safe default)
        """
        if not self.is_ready:
            logger.debug("[Spoof] Model chưa load xong → safe default REAL")
            return True, 1.0

        frame_copy = frame.copy()
        future = self._pool.submit(self._run_check, frame_copy, face)
        try:
            is_real, score = future.result(timeout=timeout)
            return is_real, score
        except FutureTimeout:
            logger.warning(f"[Spoof] Timeout {timeout*1000:.0f}ms → safe default REAL")
            return True, 1.0
        except Exception as e:
            logger.warning(f"[Spoof] check_sync error: {e} → safe default REAL")
            return True, 1.0

    # Backward compat stubs (không dùng nữa nhưng giữ để không crash nếu còn gọi)
    def submit(self, face_id, frame, face):
        pass

    def get_verdict(self, face_id, min_votes=1):
        return True, 1.0

    def cleanup(self, face_id):
        pass

    def cleanup_stale(self, active_ids):
        pass

    def shutdown(self):
        self._pool.shutdown(wait=False)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _run_check(self, frame: np.ndarray, face) -> Tuple[bool, float]:
        """Chạy trong thread pool — kết quả được future.result() chờ."""
        t0 = time.time()

        # Lớp 1: Laplacian blur score (~1ms)
        # Phone screen sắc nét (blur > 60) → tiếp tục ML
        # Ảnh in mờ / màn hình cũ (blur < 60) → FAKE ngay
        bbox = face.bbox.astype(int)
        x1   = max(0, bbox[0]);  y1 = max(0, bbox[1])
        x2   = min(frame.shape[1], bbox[2])
        y2   = min(frame.shape[0], bbox[3])

        if x2 <= x1 or y2 <= y1:
            return True, 1.0  # Invalid crop → safe default

        face_crop  = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return True, 1.0

        gray_crop  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

        # Phone màn hình OLED sắc nét thường có blur > 60 nên lớp blur
        # không bắt được → cần dựa vào ML (Lớp 2)
        BLUR_THRESH = 45.0   # Tăng lên từ 60 → bắt thêm ảnh in chất lượng trung bình

        if blur_score < BLUR_THRESH:
            score_fake = min(1.0, 1.0 - blur_score / BLUR_THRESH)
            elapsed    = (time.time() - t0) * 1000
            logger.info(f"[Spoof] ❌ FAKE (blur={blur_score:.1f} < {BLUR_THRESH}) [{elapsed:.0f}ms]")
            return False, score_fake

        # Lớp 2: MiniFASNetV2 ML model (~20-40ms)
        # Phát hiện: ảnh in HD, phone màn hình sáng, video phát lại, mặt nạ
        try:
            label, score = self._model.predict(frame, face)
            is_real      = (label == 1)
            elapsed      = (time.time() - t0) * 1000
            logger.info(
                f"[Spoof] {'✅ REAL' if is_real else '❌ FAKE'} "
                f"(ML score={score:.2f}, blur={blur_score:.0f}) [{elapsed:.0f}ms]"
            )
            return is_real, float(score)
        except Exception as e:
            logger.warning(f"[Spoof] ML predict error: {e} → safe default REAL")
            return True, 1.0
