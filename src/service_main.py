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

# --- SHARED MEMORY FOR PERSISTENT MONITORING (ULTRA STABLE) ---
from multiprocessing import shared_memory
import numpy as np

SHM_NAME = "bittech_monitor_shm"
SHM_SIZE = 5 * 1024 * 1024 # Increased to 5MB for 720p high quality

try:
    # Try to connect to existing or create new
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        logger.info("Found existing Shared Memory.")
    except FileNotFoundError:
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
        # 1. Prepare Metadata
        h, w = frame.shape[:2]
        data = frame.tobytes()
        size = len(data)
        
        if size > SHM_SIZE - 20: # Safety margin
            return
            
        # 2. Update sequence (ODD = Writing)
        current_seq = shm.buf[0]
        shm.buf[0] = (current_seq + 1) % 255
        
        # 3. Write Format (Offset 1): 1 = RAW
        shm.buf[1] = 1
        # Write W, H (Offset 2, 4)
        shm.buf[2:4] = np.array([w], dtype=np.uint16).tobytes()
        shm.buf[4:6] = np.array([h], dtype=np.uint16).tobytes()
        
        # 4. Write Pixel Data (Offset 10)
        shm.buf[10:10+size] = data
        
        # 5. Done (EVEN = Valid)
        shm.buf[0] = (current_seq + 2) % 255
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
    # AI chỉ xử lý 1 trong 5 frame → tại 15fps = 3 lần/giây
    # Giảm CPU rất đáng kể mà không ảnh hưởng tốc độ nhận diện thực tế
    process_every_n_frames = 5
    faces = []
    ai_busy = False
    last_preview_time = 0
    # Giới hạn preview FPS theo camera FPS thực tế (tối đa 30)
    preview_fps = min(30, CameraConfig.FPS)
    preview_interval = 1.0 / preview_fps

    try:
        from src.attendance.mongodb_mgr import mongo_db
        CameraConfig.load_from_mongodb(mongo_db)
        
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.0)
        
        # Notify about successful startup
        from src.utils.notification import show_info_message
        show_info_message("Bittech Camera Service", "DỊCH VỤ CAMERA ĐÃ BẬT THÀNH CÔNG!\n\nHệ thống đang chạy ẩn và sẽ tự động điểm danh.")
        
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
                    roi=CameraConfig.ROI  # None → full frame, tuple → crop trước detect
                )
                tracker.update(detected, attendance, face_rec=face_rec, frame=ai_frame)

                
                # Update shared list
                faces[:] = detected 
            except Exception as e:
                logger.error(f"AI Worker error: {e}")
            finally:
                ai_queue.task_done()

    threading.Thread(target=ai_worker_persistent, daemon=True).start()

    # --- PREVIEW WORKER: draw + HUD + resize + SHM trong thread riêng ---
    # Main thread chỉ push (frame, faces snapshot) → preview thread xử lý async
    # → Camera read loop không bao giờ bị block bởi annotation/SHM
    preview_queue = queue.Queue(maxsize=1)

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

                # FPS smooth
                dt = ts - last_ts if ts > last_ts else 0.001
                last_ts = ts
                fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / dt)

                # Draw bounding boxes
                annotated = face_rec.draw_faces(pframe, pfaces) if pfaces else pframe

                # ROI box
                if CameraConfig.ROI:
                    x1, y1, x2, y2 = CameraConfig.ROI
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,220,220), 2)
                    _shadow(annotated, "VUNG CHAM CONG", (x1+8, y1+26), 0.6, (0,220,220), 2)

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

                # Thu nhỏ preview → SHM (640×360 = 675KB, SHM 5MB ok)
                preview_frame = cv2.resize(annotated, (640, 360))
                write_frame_to_shm(preview_frame)

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

            # 2. READ FRAME
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue

            # 3. PUSH FRAME TO PREVIEW THREAD (non-blocking, ≤ SHM write cost)
            # Preview worker sẽ draw + HUD + write SHM — không block camera read
            if not preview_queue.full():
                try:
                    preview_queue.put_nowait((frame.copy(), list(faces), current_time))
                except Exception:
                    pass

            # 4. ASYNC AI TRIGGER — chỉ đẩy frame vào queue mỗi N vòng lặp
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                if not ai_queue.empty():
                    try:
                        ai_queue.get_nowait()
                    except Exception:
                        pass
                try:
                    ai_queue.put_nowait(frame.copy())
                except Exception:
                    pass
            # NOTE: Không có time.sleep() — RTSP read() tự block chờ frame mới từ camera

    except KeyboardInterrupt:
        logger.info("Service stopping...")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        camera.disconnect()
        logger.info("Service shutdown complete.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
