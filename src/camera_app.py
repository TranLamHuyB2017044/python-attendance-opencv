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

# ── Webhook log panel constants ──────────────────────────────────────────────
WEBHOOK_API_URL  = "https://voice-cheking.bittechx.cloud/api/users"
LOG_POLL_INTERVAL = 3.0          # seconds between API polls
LOG_PANEL_RATIO   = 0.25         # 25 % of total window width
LOG_MAX_ROWS      = 30           # max rows kept in memory

# Status → (BGR color, icon char)
STATUS_META = {
    "IN":       ((0, 220, 80),   "+"),
    "OUT":      ((60, 100, 255), "-"),
    "COOLDOWN": ((0, 200, 255),  "~"),
    "unknown":  ((80,  80,  80), "?"),
}


# ── Shared log state (thread-safe via a lock) ─────────────────────────────────
_log_lock    = threading.Lock()
_log_entries = []          # list of dicts from API (newest first, max LOG_MAX_ROWS)
_log_newest_id = -1        # highest log id seen so far (for blink highlight)
_log_new_ids   = set()     # ids that arrived in the LAST poll cycle


def _log_fetch_worker():
    """Background thread: poll WEBHOOK_API_URL every LOG_POLL_INTERVAL seconds."""
    global _log_entries, _log_newest_id, _log_new_ids

    while True:
        try:
            req = urllib.request.Request(
                WEBHOOK_API_URL,
                headers={"User-Agent": "BitTech-AttendanceMonitor/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if isinstance(data, list) and data:
                # Determine which are truly new since last poll
                with _log_lock:
                    old_max = _log_newest_id
                new_ids_this_cycle = set()
                new_max = old_max

                for entry in data:
                    eid = entry.get("id", -1)
                    if eid > old_max:
                        new_ids_this_cycle.add(eid)
                        if eid > new_max:
                            new_max = eid

                with _log_lock:
                    _log_entries  = data[:LOG_MAX_ROWS]
                    _log_newest_id = new_max
                    _log_new_ids   = new_ids_this_cycle

        except Exception as exc:
            logger.debug(f"Log fetch error: {exc}")

        time.sleep(LOG_POLL_INTERVAL)


def _draw_log_panel(canvas: np.ndarray, x_off: int, panel_w: int, panel_h: int):
    """
    Render a dark side-panel onto `canvas` starting at column x_off.
    Shows the latest webhook attendance records in real-time.
    """
    # ── Background ────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (x_off, 0), (x_off + panel_w, panel_h),
                  (18, 18, 28), -1)          # very dark navy
    # Thin separator line
    cv2.line(canvas, (x_off, 0), (x_off, panel_h), (50, 50, 80), 2)

    # ── Header ────────────────────────────────────────────────────────────────
    header_h = 48
    cv2.rectangle(canvas, (x_off, 0), (x_off + panel_w, header_h),
                  (28, 28, 48), -1)
    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    # Title
    cv2.putText(canvas, "WEBHOOK LOG", (x_off + 8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, now_str, (x_off + 8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 160, 200), 1, cv2.LINE_AA)

    # Live dot (blink every second)
    dot_color = (0, 230, 80) if int(time.time()) % 2 == 0 else (0, 120, 40)
    cv2.circle(canvas, (x_off + panel_w - 14, 20), 5, dot_color, -1)

    # ── Entries ───────────────────────────────────────────────────────────────
    with _log_lock:
        entries    = list(_log_entries)
        new_ids    = set(_log_new_ids)

    row_h  = 54          # height per log row
    y_base = header_h + 6
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for i, entry in enumerate(entries):
        y_top = y_base + i * row_h
        if y_top + row_h > panel_h:
            break

        eid    = entry.get("id", -1)
        status = entry.get("status", "unknown")
        name   = entry.get("user_name", "Unknown")
        uid    = str(entry.get("user_id", ""))
        t_str  = entry.get("time", "")
        vtxt   = entry.get("voice_text", "")

        color, icon = STATUS_META.get(status, ((120, 120, 120), "?"))
        is_new = eid in new_ids

        # Row background (highlight new entries briefly)
        row_bg = (35, 35, 55) if not is_new else (35, 55, 35)
        cv2.rectangle(canvas, (x_off + 2, y_top),
                      (x_off + panel_w - 2, y_top + row_h - 2), row_bg, -1)

        # Status pill
        pill_x = x_off + 6
        pill_y = y_top + 8
        pill_w, pill_h = 42, 18
        cv2.rectangle(canvas, (pill_x, pill_y),
                      (pill_x + pill_w, pill_y + pill_h), color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{icon} {status[:4]}", (pill_x + 2, pill_y + 13),
                    font, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # Time
        cv2.putText(canvas, t_str, (x_off + pill_w + 12, y_top + 20),
                    font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Name (truncate if too long)
        max_name_w = panel_w - 14
        name_disp = name if len(name) <= 22 else name[:20] + ".."
        cv2.putText(canvas, name_disp, (x_off + 6, y_top + 38),
                    font, 0.42, (220, 230, 255), 1, cv2.LINE_AA)

        # UID small
        uid_disp = f"#{uid}" if uid and uid != "Unknown" else ""
        if uid_disp:
            cv2.putText(canvas, uid_disp, (x_off + panel_w - 70, y_top + 38),
                        font, 0.30, (100, 120, 150), 1, cv2.LINE_AA)

        # Divider
        cv2.line(canvas, (x_off + 4, y_top + row_h - 1),
                 (x_off + panel_w - 4, y_top + row_h - 1), (40, 40, 60), 1)

    # ── Footer: total count ───────────────────────────────────────────────────
    with _log_lock:
        cnt = len(_log_entries)
    cv2.putText(canvas, f"Total records: {cnt}", (x_off + 8, panel_h - 8),
                font, 0.33, (70, 90, 120), 1, cv2.LINE_AA)


def main():
    setup_logger()
    warnings.filterwarnings("ignore", category=FutureWarning)
    logger.info("Starting Auto-Start Camera Detection...")

    # Start Ping Service
    ping_service.start()

    # ── Start log-fetch background thread (TEST_MODE only) ──────────────────
    _test_mode = RecognitionConfig.TEST_MODE
    if _test_mode:
        log_thread = threading.Thread(target=_log_fetch_worker, daemon=True)
        log_thread.start()
        logger.info(f"[TEST_MODE] Webhook log panel started — polling {WEBHOOK_API_URL}")
    else:
        logger.info("Webhook log panel disabled (TEST_MODE=false)")

    try:
        face_rec   = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera     = RTSPCamera()
        tracker    = FaceTracker(threshold_seconds=2.0)
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        return

    # ── Window setup ──────────────────────────────────────────────────────────
    TOTAL_W = 1600 if _test_mode else 1280
    TOTAL_H = 900  if _test_mode else 720
    win_name = "CAMERA TU DONG DIEM DANH"
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
            if _test_mode:
                panel_w = max(200, int(cur_w * LOG_PANEL_RATIO))
                cam_w   = cur_w - panel_w
            else:
                panel_w = 0
                cam_w   = cur_w

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
            canvas[y_off:y_off + new_cam_h, 0:new_cam_w] = cam_resized

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

            # ── RIGHT: Log panel (25%) — TEST_MODE only ──────────────────
            if _test_mode and panel_w > 0:
                _draw_log_panel(canvas, cam_w, panel_w, cur_h)

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
