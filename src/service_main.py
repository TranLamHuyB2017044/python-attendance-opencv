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
                    
                    # Create a "waiting" preview so management app doesn't freeze
                    try:
                        import numpy as np
                        waiting_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(waiting_frame, "DANG CHO KET NOI CAMERA...", (120, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        cv2.putText(waiting_frame, f"Retrying: {time.strftime('%H:%M:%S')}", (200, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        cv2.putText(waiting_frame, "Service is running but camera offline", (140, 240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                        cv2.imwrite(str(preview_path), waiting_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    except Exception as e:
                        logger.error(f"Failed to create waiting preview: {e}")
                    
                    time.sleep(10) # Wait before retry
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
                
                # 2. Save a small preview frame
                small_frame = cv2.resize(annotated_preview, (640, 360))
                # Add a "LIVE" indicator
                cv2.putText(small_frame, f"LIVE MONITOR: {time.strftime('%H:%M:%S')}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(str(preview_path), small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
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
