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

def main():
    """
    Headless background service for face recognition and attendance logging.
    Designed to run without UI (no cv2.imshow).
    """
    setup_logger()
    logger.info("Starting Background Face Recognition Service...")

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.0)
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        return

    logger.info("System initialized. Processing camera feed...")
    
    from src.config import DATA_DIR
    import cv2
    import os
    
    preview_path = DATA_DIR / "camera_preview.jpg"

    try:
        while True:
            if not camera.is_connected:
                if not camera.connect():
                    time.sleep(10) # Wait before retry
                    continue
            
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue

            # Process frame
            faces = face_rec.detect_and_extract(frame)
            tracker.update(faces, attendance, frame)
            
            # --- NEW: UPDATE HEARTBEAT & PREVIEW ---
            try:
                # 1. Update heartbeat in MongoDB (Company specific)
                from src.config import MongoDbConfig
                from src.attendance.mongodb_mgr import mongo_db
                mongo_db.db.system_status.update_one(
                    {"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID},
                    {"$set": {"last_seen": time.time(), "status": "running"}},
                    upsert=True
                )
                
                # 2. Draw results for preview
                annotated_preview = face_rec.draw_faces(frame, faces)
                
                # 3. Save a small preview frame
                small_frame = cv2.resize(annotated_preview, (640, 360))
                # Add a "LIVE" indicator
                cv2.putText(small_frame, f"LIVE MONITOR: {time.strftime('%H:%M:%S')}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(str(preview_path), small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            except Exception as e:
                logger.warning(f"Failed to update service status: {e}")

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
