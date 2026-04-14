import cv2
import time
import warnings
import numpy as np
import sys
import os
import threading
import queue
import urllib.request
import json
import datetime
from loguru import logger

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.camera.rtsp_camera import RTSPCamera
from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.recognition.tracker import FaceTracker
from src.config import RecognitionConfig, CameraConfig
from src.services.ping_service import ping_service

from src.utils.log_panel_renderer import draw_log_panel
from src.utils.webhook_log_bus import start_polling

# ══════════════════════════════════════════════════════════════════════════════
#  WEBHOOK LOG PANEL
# ══════════════════════════════════════════════════════════════════════════════
_LOG_PANEL_RATIO = 0.25          # 25% chiều rộng màn hình

# ── Status colors / short labels ─────────────────────────────────────────────



def main():
    setup_logger()
    warnings.filterwarnings("ignore", category=FutureWarning)
    logger.info("Starting Auto-Start Camera Detection...")

    # Start Ping Service
    ping_service.start()

    # ── Start log-fetch background thread ────────────────────────────────────
    start_polling()

    try:
        face_rec   = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera     = RTSPCamera()
        tracker    = FaceTracker(threshold_seconds=2.0)
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        return

    # ── Window setup ──────────────────────────────────────────────────────────
    TOTAL_W, TOTAL_H = 1600, 900
    win_name = "He thong Diem danh Khuon mat"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, TOTAL_W, TOTAL_H)

    faces = []
    ai_queue = queue.Queue(maxsize=1)
    frame_skip_count = 0
    PROCESS_EVERY_N  = 3

    def ai_worker():
        nonlocal frame_skip_count
        while True:
            try:
                ai_frame = ai_queue.get()
                if ai_frame is None:
                    break

                frame_skip_count += 1
                if frame_skip_count % PROCESS_EVERY_N != 0:
                    ai_queue.task_done()
                    continue

                detected = face_rec.detect_and_extract(ai_frame, max_faces=RecognitionConfig.MAX_FACES)
                tracker.update(detected, attendance, frame=ai_frame, face_rec=face_rec)
                faces[:] = detected
            except Exception as e:
                logger.error(f"AI Worker Error: {e}")
            finally:
                ai_queue.task_done()

    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    logger.info("Press 'q' to exit camera.")

    def put_shadow(img, text, pos, scale, color, thick):
        """Text with 1-px black shadow for legibility on any background."""
        cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1)
        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    # --- Mouse Callback logic for Log Panel Buttons ---
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Lấy kích thước hiện tại từ window để tính toán click area
            try:
                _, _, cw, ch = cv2.getWindowImageRect(win_name)
                if cw > 0:
                    pw = max(220, int(cw * _LOG_PANEL_RATIO))
                    x0 = cw - pw
                    if x >= x0 + pw - 55 and y >= ch - 22:
                        from src.utils.webhook_log_bus import clear_logs
                        clear_logs()
            except Exception: pass

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse_click)
    # ───────────────────────────────────────────────────

    try:
        while True:
            # ── Camera connect loop ──────────────────────────────────────────
            if not camera.is_connected:
                if not camera.connect():
                    logger.warning("Camera not connected. Retrying in 5s...")
                    time.sleep(5)
                    continue

            success, frame = camera.read_frame()
            if not success or frame is None:
                continue

            # ── Feed AI thread ───────────────────────────────────────────────
            if not ai_queue.empty():
                try:
                    ai_queue.get_nowait()
                except Exception:
                    pass

            try:
                ai_frame_obj = frame.copy()
                if CameraConfig.ROI:
                    x1, y1, x2, y2 = CameraConfig.ROI
                    mask = np.zeros_like(ai_frame_obj)
                    h, w = ai_frame_obj.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    mask[y1:y2, x1:x2] = 255
                    ai_frame_obj = cv2.bitwise_and(ai_frame_obj, mask)
                ai_queue.put_nowait(ai_frame_obj)
            except queue.Full:
                pass

            # ── Get current window dimensions ────────────────────────────────
            try:
                _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
                if cur_w <= 0 or cur_h <= 0:
                    cur_w, cur_h = TOTAL_W, TOTAL_H
            except Exception:
                cur_w, cur_h = TOTAL_W, TOTAL_H

            # In TEST_MODE: 75% cam + 25% log panel; else full width
            panel_w = max(220, int(cur_w * _LOG_PANEL_RATIO))
            cam_w   = cur_w - panel_w

            # ── Compose display frame ────────────────────────────────────────
            canvas = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)

            # --- LEFT: Camera view (75%) ---
            cam_display = face_rec.draw_faces(frame.copy(), faces)

            if CameraConfig.ROI:
                x1, y1, x2, y2 = CameraConfig.ROI
                cv2.rectangle(cam_display, (x1, y1), (x2, y2), (0, 230, 230), 2)
                cv2.putText(cam_display, "VUNG CHAM CONG", (x1 + 10, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2)

            # Resize camera view to fit left portion
            h_f, w_f = cam_display.shape[:2]
            scale_ratio = min(cam_w / w_f, cur_h / h_f)
            new_cam_w = int(w_f * scale_ratio)
            new_cam_h = int(h_f * scale_ratio)
            cam_resized = cv2.resize(cam_display, (new_cam_w, new_cam_h))

            # Center vertically in left region
            y_off = (cur_h - new_cam_h) // 2
            x_off = (cam_w - new_cam_w) // 2
            canvas[y_off:y_off + new_cam_h, x_off:x_off + new_cam_w] = cam_resized

            # ── HUD overlay on camera region ──────────────────────────────────
            now_str      = datetime.datetime.now().strftime("%H:%M:%S")
            current_time = time.time()

            fps = 0.0
            if hasattr(camera, 'last_frame_time') and camera.last_frame_time:
                dt = current_time - camera.last_frame_time
                fps = 1.0 / dt if dt > 0 else 0.0
            camera.last_frame_time = current_time

            ping_str = getattr(camera, 'current_ping', 'N/A')
            try:
                ping_ms = int(ping_str.replace('ms', '')) if 'ms' in ping_str else -1
            except Exception:
                ping_ms = -1

            fps_color  = (0, 230, 0) if fps  >= 12 else (0, 200, 255) if fps  >= 7 else (0, 60, 255)
            ping_color = (0, 230, 0) if ping_ms < 50 else (0, 200, 255) if ping_ms < 150 else (0, 60, 255)
            if ping_ms < 0:
                ping_color = (160, 160, 160)

            put_shadow(canvas, f"FPS {int(fps)}", (10, 26), 0.65, fps_color,  2)
            put_shadow(canvas, f"| {ping_str}",   (95, 26), 0.65, ping_color, 2)
            put_shadow(canvas, now_str, (cam_w - 95, 26), 0.55, (200, 200, 200), 1)

            face_count = len(faces)
            if face_count > 0:
                put_shadow(canvas, f"Faces: {face_count}",
                           (10, cur_h - 12), 0.5, (160, 220, 160), 1)

            put_shadow(canvas, "[Q] Thoat", (10, cur_h - 30), 0.45, (160, 160, 160), 1)

            # --- RIGHT: Log panel (25%) ---
            if panel_w > 0:
                draw_log_panel(canvas, cam_w, panel_w, cur_h)

            cv2.imshow(win_name, canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Stopping camera...")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
        ping_service.stop()
        logger.info("Camera app shut down.")

if __name__ == "__main__":
    main()
