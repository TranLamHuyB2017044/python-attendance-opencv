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
    process_every_n_frames = 3
    faces = []
    ai_busy = False
    last_preview_time = 0
    # Match camera FPS for preview (limit to 15-25 for stability)
    preview_fps = max(15, min(25, CameraConfig.FPS))
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
                
                # Heavy AI Tasks
                detected = face_rec.detect_and_extract(ai_frame)
                tracker.update(detected, attendance, ai_frame)
                
                # Update shared list
                faces[:] = detected 
            except Exception as e:
                logger.error(f"AI Worker error: {e}")
            finally:
                ai_queue.task_done()

    threading.Thread(target=ai_worker_persistent, daemon=True).start()
    logger.info("System initialized. Processing camera at high speed...")

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

            # 3. PRIORITY PREVIEW (Syncing boxes with correct resolution)
            if current_time - last_preview_time >= preview_interval:
                try:
                    # DRAW FIRST on original resolution to keep boxes correct
                    if faces:
                        annotated_full = face_rec.draw_faces(frame.copy(), faces)
                    else:
                        annotated_full = frame
                    
                    # THEN RESIZE for monitor performance
                    preview_frame = cv2.resize(annotated_full, (960, 540))
                    
                    write_frame_to_shm(preview_frame)
                    last_preview_time = current_time
                except: pass

            # 4. ASYNC AI TRIGGER (Queue based)
            if ai_queue.empty():
                try:
                    ai_queue.put_nowait(frame.copy())
                except: pass

            time.sleep(0.001) 

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
