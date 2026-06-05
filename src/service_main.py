import cv2
import time
import warnings
import sys
import os
from loguru import logger

# --- CẤU HÌNH ĐƯỜNG DẪN CHO PYINSTALLER ---
# Thêm thư mục gốc vào danh sách tìm kiếm module
# --- FIX FOR WINDOWED MODE (NoneType.write error) ---
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

if getattr(sys, 'frozen', False):
    # Nếu chạy từ file .exe
    base_dir = sys._MEIPASS
    # --- THÊM ĐƯỜNG DẪN NGOÀI ĐỂ HỖ TRỢ LIVE UPDATE (Dành cho Loader + Source flow) ---
    # Cho phép ghi đè logic bằng cách copy file .py vào thư mục 'src' bên cạnh file .exe
    exe_dir = os.path.dirname(sys.executable)
    if exe_dir not in sys.path:
        sys.path.insert(0, exe_dir) 
else:
    # Nếu chạy từ code python
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)
# ------------------------------------------

from src.utils.logger import setup_logger
from src.camera.rtsp_camera import RTSPCamera
from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.recognition.tracker import FaceTracker

from src.utils.notification import send_notification, show_error_message
from src.services.ping_service import ping_service

# --- SHARED MEMORY FOR PERSISTENT MONITORING (ULTRA STABLE) ---
from multiprocessing import shared_memory
import numpy as np

SHM_NAME = "bittech_monitor_shm"
SHM_SIZE = 20 * 1024 * 1024 # Increased to 20MB to support up to 4K resolution (approx 2.7MB for 720p, 6.2MB for 1080p)

try:
    # Try to connect to existing or create new
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        # Verify size if it's already there (Optional, but let's be safe)
        if shm.size < SHM_SIZE:
            logger.info(f"Existing SHM too small ({shm.size} < {SHM_SIZE}), recreating...")
            shm.close()
            # On Windows, we might need some trick to truly 'delete' it or just recreate with new name if it persists
            # But usually it stays until all handles close.
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        else:
            logger.info("Found existing Shared Memory.")
    except Exception:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        logger.success("Created new Shared Memory for monitoring.")
except Exception as e:
    logger.error(f"Failed to init Shared Memory: {e}")
    shm = None

def write_frame_to_shm(frame):
    """Writes RAW pixels into Shared Memory for zero-CPU display (Ultra smooth)."""
    global shm
    if shm is None or frame is None: return
    try:
        h, w = frame.shape[:2]
        data = frame.tobytes()
        size = len(data)

        if size > SHM_SIZE - 20:
            if int(time.time()) % 10 == 0:
                logger.warning(f"Frame too large for SHM ({size} > {SHM_SIZE}). Skipping write.")
            return

        # Protocol: ODD seq = đang ghi (reader BỎ QUA), EVEN seq = đã xong (reader ĐỌC)
        # % 256 (ĐÚNG): khi current=254 → write=255(ODD✅) → done=0(EVEN✅)
        # % 255 (SAI):  khi current=254 → write=0(EVEN❌)  → done=1(ODD❌)  ← bug cũ
        current_seq = int(shm.buf[0])
        shm.buf[0] = (current_seq + 1) % 256   # ODD = đang ghi
        
        shm.buf[1] = 1  # format = RAW
        shm.buf[2:4] = np.array([w], dtype=np.uint16).tobytes()
        shm.buf[4:6] = np.array([h], dtype=np.uint16).tobytes()
        shm.buf[10:10 + size] = data

        shm.buf[0] = (current_seq + 2) % 256   # EVEN = ghi xong, reader được đọc
    except Exception as e:
        logger.debug(f"SHM Write error: {e}")


# Note: FastAPI server is no longer needed but kept as empty if you want to reuse it later
# or we can just remove uvicorn to be clean.
# -------------------------------------------------------------

def main():
    """
    Headless background service for face recognition and attendance logging.
    Designed to run without UI (no cv2.imshow).
    """
    setup_logger()
    logger.info("Starting Background Face Recognition Service...")

    # Chặn mở nhiều service cùng lúc
    from src.utils.single_instance import force_single_instance
    force_single_instance("CameraService")

    # --- INITIALIZE VARIABLES ---
    from src.config import DATA_DIR, CameraConfig
    import cv2
    import os
    import threading
    
    preview_path = DATA_DIR / "camera_preview.jpg"
    frame_count = 0
    # Time-based AI throttle: tối đa 3 lần/giây (333ms giữa 2 lần detect)
    # Đảm bảo mẫu số >= 1 để tránh lỗi ZeroDivisionError
    DETECT_PER_SEC = 3
    AI_INTERVAL  = 1.0 / max(1, DETECT_PER_SEC) 
    last_ai_time = 0.0
    faces = []
    ai_busy = False
    last_preview_time = 0
    # Đã chuyển sang dùng worker cho preview
    
    settings_watcher = None
    try:
        from src.attendance.mongodb_mgr import mongo_db
        CameraConfig.load_from_mongodb(mongo_db)
        
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.0)
        
        # Start background settings watcher to support hot-reloading (Change Streams or Polling fallback)
        from src.services.settings_watcher import SettingsWatcher
        settings_watcher = SettingsWatcher(camera)
        settings_watcher.start()
        
        # Startup logic moved into loop to ensure notifications only sent after CAMERA success
        system_initialized_notified = False

        logger.info("Service is now running in the background.")
        
    except Exception as e:
        error_msg = f"Lỗi khởi tạo: {str(e)}"
        logger.error(error_msg)
        show_error_message("Lỗi Dịch Vụ AI", f"Không thể khởi động dịch vụ AI:\n{error_msg}")
        return

    # --- NON-BLOCKING HEARTBEAT THREAD ---
    def heartbeat_loop():
        from src.attendance.mongodb_mgr import mongo_db
        from src.config import MongoDbConfig
        while True:
            try:
                cam_status = "running" if camera.is_connected else "waiting_camera"
                mongo_db.db.system_status.update_one(
                    {"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID},
                    {"$set": {
                        "last_seen": time.time(), 
                        "status": cam_status,
                        "camera_connected": camera.is_connected
                    }},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
            time.sleep(5)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    # --- PERSISTENT AI WORKER ---
    import queue
    ai_queue = queue.Queue(maxsize=1) 
    
    def ai_worker_persistent():
        while True:
            try:
                ai_frame = ai_queue.get()
                if ai_frame is None: break
                
                # Heavy AI Tasks (Detection only in fast mode, ROI crop nếu cấu hình)
                detected = face_rec.detect_and_extract(
                    ai_frame, fast=True,
                    roi=CameraConfig.ROI
                )
                tracker.update(detected, attendance, face_rec=face_rec, frame=ai_frame)
                faces[:] = detected

                # Nhưỡng CPU 50ms sau mỗi inference — giảm CPU spike
                # ONNX đã chạy xong, 50ms nghỉ không ảnh hưởng độ trễ detect
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"AI Worker error: {e}")
            finally:
                ai_queue.task_done()

    threading.Thread(target=ai_worker_persistent, daemon=True).start()

    # --- PREVIEW WORKER: draw + HUD + resize + SHM trong thread riêng ---
    # Main thread chỉ push (frame, faces snapshot) → preview thread xử lý async
    # → Camera read loop không bao giờ bị block bởi annotation/SHM
    preview_queue = queue.Queue(maxsize=1)
    last_preview_push = 0.0
    # Đảm bảo mẫu số >= 1 để tránh lỗi ZeroDivisionError
    PREVIEW_TARGET_FPS = 15
    PREVIEW_INTERVAL  = 1.0 / max(1, PREVIEW_TARGET_FPS) 

    def preview_worker():
        import datetime, numpy as np
        last_ts = time.time()
        fps_smooth = 0.0

        def _shadow(img, text, pos, scale, color, thick):
            cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1)
            cv2.putText(img, text, pos,              cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thick)

        while True:
            try:
                item = preview_queue.get()
                if item is None:
                    break
                pframe, pfaces, ts = item

                # FPS smooth (dựa trên timestamp frame từ main loop)
                now = time.time()
                # Bảo vệ dt cực kỳ nghiêm ngặt chống ZeroDivisionError
                dt = now - last_ts
                if dt <= 0: dt = 0.001 
                
                last_ts = now
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

                # Draw bounding boxes (draw_faces tự copy frame nội bộ — thread-safe)
                # pfaces đã được copy ở thread chính
                annotated = face_rec.draw_faces(pframe.copy(), pfaces) if pfaces else pframe.copy()

                # ROI box
                if CameraConfig.ROI:
                    x1, y1, x2, y2 = CameraConfig.ROI
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (255, 120, 0), 2)
                    _shadow(annotated, "VUNG CHAM CONG", (x1+8, y1+26), 0.6, (255, 120, 0), 2)

                # HUD
                h_f, w_f = annotated.shape[:2]
                ping_str  = getattr(camera, 'current_ping', 'N/A')
                try:
                    ping_ms = int(ping_str.replace('ms','')) if 'ms' in ping_str else -1
                except Exception:
                    ping_ms = -1
                fps       = fps_smooth
                fps_color  = (0,230,0) if fps  >= 12 else (0,200,255) if fps  >= 7 else (0,60,255)
                ping_color = (0,230,0) if ping_ms < 50 else (0,200,255) if ping_ms < 150 else (0,60,255)
                if ping_ms < 0: ping_color = (160,160,160)

                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                _shadow(annotated, f"FPS {int(fps)}", (10,26),      0.65, fps_color,    2)
                _shadow(annotated, f"| {ping_str}",   (95,26),      0.65, ping_color,   2)
                _shadow(annotated, now_str,           (w_f-95,26),  0.55, (200,200,200),1)
                if len(pfaces) > 0:
                    _shadow(annotated, f"Faces: {len(pfaces)}", (10,h_f-12), 0.5, (160,220,160), 1)

                # Ghi full resolution 1280×720 vào SHM — sắc nét tương đương camera_app.py
                # An toàn vì: (1) FPS limiter 15fps, (2) Management App đã bỏ AI (~30% CPU giảm)
                # 1280×720×3 = 2.76MB < 5MB SHM limit ✅
                write_frame_to_shm(annotated)

            except Exception as e:
                logger.debug(f"[PreviewWorker] {e}")
            finally:
                preview_queue.task_done()

    threading.Thread(target=preview_worker, daemon=True, name="preview_worker").start()
    logger.info("System initialized. Camera service running (preview offloaded to thread)...")

    try:
        while True:
            current_time = time.time()
            
            # 1. CAMERA CHECK
            if not camera.is_connected:
                camera.connect() 
                if not camera.is_connected:
                    time.sleep(1)
                    continue
                
                # --- AUTO NOTIFY ONCE AFTER FIRST SUCCESSFUL STARTUP ---
                if not system_initialized_notified:
                    from src.utils.notification import send_notification
                    send_notification("Bittech Camera Service", "DỊCH VỤ CAMERA ĐÃ HOẠT ĐỘNG!\n\nHệ thống bắt đầu quét khuôn mặt.")
                    # ping_service.start() # Bắt đầu gửi Ping khi đã có hình ảnh
                    system_initialized_notified = True
                    logger.success("Service reporting started after camera connection.")

            # 2. READ FRAME
            # wait_first_frame=True: cho background thread co it nhat 1 frame
            # truoc khi service loop bat dau push vao preview_queue
            success, frame = camera.read_frame(wait_first_frame=True)
            if not success or frame is None:
                time.sleep(0.005)  # tranh busy-spin khi khong co frame
                continue

            # 3. PUSH FRAME TO PREVIEW THREAD (FPS-limited + non-blocking)
            # FPS limiter ở đây thay vì trong worker → worker LUÔN ghi SHM khi nhận frame
            if current_time - last_preview_push >= PREVIEW_INTERVAL:
                if not preview_queue.full():
                    try:
                        # Copy list faces để tránh race condition khi AI thread update cùng lúc
                        faces_copy = list(faces)  # shallow copy đủ an toàn với list replace
                        preview_queue.put_nowait((frame, faces_copy, current_time))
                        last_preview_push = current_time
                    except Exception:
                        pass

            # 4. TIME-BASED AI TRIGGER — tối đa 3 lần/giây bất kể FPS camera
            if (current_time - last_ai_time) >= AI_INTERVAL:
                if not ai_queue.empty():
                    try: ai_queue.get_nowait()
                    except Exception: pass
                try:
                    ai_queue.put_nowait(frame.copy())  # copy một lần duy nhất
                    last_ai_time = current_time
                except Exception:
                    pass

            # Safety yield: tránh busy-spin khi RTSP buffer đầy
            time.sleep(0.002)

    except KeyboardInterrupt:
        logger.info("Service stopping...")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        ping_service.stop()  # Dừng Ping Service khi camera service tắt
        if settings_watcher:
            try:
                settings_watcher.stop()
            except Exception:
                pass
        camera.disconnect()
        logger.info("Service shutdown complete.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
