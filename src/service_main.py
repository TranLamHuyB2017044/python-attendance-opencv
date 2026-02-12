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
    """Writes JPEG bytes into the Shared Memory buffer with a sequence counter."""
    global shm
    if shm is None or frame is None: return
    try:
        # 1. Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret: return
        
        data = buffer.tobytes()
        size = len(data)
        
        if size > SHM_SIZE - 10:
            return
            
        # 2. Get current sequence and increment to ODD (Signals "Work in Progress")
        current_seq = shm.buf[0]
        shm.buf[0] = (current_seq + 1) % 255
        
        # 3. Write Size (offset 1) and Data (offset 5)
        shm.buf[1:5] = np.array([size], dtype=np.uint32).tobytes()
        shm.buf[5:5+size] = data
        
        # 4. Increment to EVEN (Signals "Done / Valid Data")
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

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.0)
        
        # Notify about successful startup
        from src.utils.notification import show_info_message
        show_info_message("Bittech Camera Service", "DỊCH VỤ CAMERA ĐÃ BẬT THÀNH CÔNG!\n\nHệ thống đang chạy ẩn và sẽ tự động điểm danh.")
        
        # Log version or info
        logger.info("Service is now running in the background.")
        
    except Exception as e:
        error_msg = f"Lỗi khởi tạo: {str(e)}"
        logger.error(error_msg)
        show_error_message("Lỗi Dịch Vụ AI", f"Không thể khởi động dịch vụ AI:\n{error_msg}")
        return

    logger.info("System initialized. Processing camera feed...")
    
    from src.config import DATA_DIR
    import cv2
    import os
    
    preview_path = DATA_DIR / "camera_preview.jpg"

    # --- FRAME SKIPPING LOGIC ---
    frame_count = 0
    process_every_n_frames = 3 # Only run AI every 3 frames
    faces = []

    try:
        while True:
            current_time = time.time()
            
            # 1. UPDATE HEARTBEAT (Once every loop is fine, or simple interval)
            try:
                from src.config import MongoDbConfig
                from src.attendance.mongodb_mgr import mongo_db
                
                # Update heartbeat in MongoDB
                cam_status = "running" if camera.is_connected else "waiting_camera"
                mongo_db.db.system_status.update_one(
                    {"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID},
                    {"$set": {
                        "last_seen": current_time, 
                        "status": cam_status,
                        "camera_connected": camera.is_connected
                    }},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"Failed to update status: {e}")
            
            # 2. CAMERA CONNECTION LOGIC
            if not camera.is_connected:
                if not camera.connect():
                    logger.warning("Camera not connected. Creating waiting preview...")
                    try:
                        import numpy as np
                        waiting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(waiting_frame, "MAT KET NOI CAMERA (RTSP ERROR)", (350, 320), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(waiting_frame, f"Dang thu lai: {time.strftime('%H:%M:%S')}", (480, 380), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                        
                        # Ghi vào cả File và RAM để App quản lý nhận được
                        write_frame_to_shm(waiting_frame)
                        cv2.imwrite(str(preview_path), waiting_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    except Exception as e:
                        logger.error(f"Failed to create waiting preview: {e}")
                    
                    time.sleep(5) # Giảm xuống 5s để phản hồi nhanh hơn
                    continue
            
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue

            frame_count += 1
            
            # 1. AI Detection (Run only every N frames to save CPU)
            if frame_count % process_every_n_frames == 0:
                faces = face_rec.detect_and_extract(frame)
            
            # 2. Logic & Tracking (Always run)
            tracker.update(faces, attendance, frame)
            
            # --- UPDATE PREVIEW WITH ACTUAL CAMERA FEED ---
            try:
                # 1. Draw results for preview
                annotated_preview = face_rec.draw_faces(frame, faces)
                
                # 2. Prepare high-quality preview (1280x720 for Full Screen)
                preview_frame = cv2.resize(annotated_preview, (1280, 720))
                
                # Update Shared Memory for Monitoring (Ultra stable local IPC)
                write_frame_to_shm(preview_frame.copy())
                
                # Still write to file as fallback for some parts of the system
                cv2.imwrite(str(preview_path), preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            except Exception as e:
                logger.warning(f"Failed to update preview: {e}")

            # Small sleep to manage CPU usage
            time.sleep(0.01)

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
