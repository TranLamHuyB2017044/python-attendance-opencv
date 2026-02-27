import cv2
import time
import warnings
import numpy as np
import sys
import os
from loguru import logger

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.camera.rtsp_camera import RTSPCamera
from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.recognition.tracker import FaceTracker
from src.config import RecognitionConfig

def main():
    setup_logger()
    warnings.filterwarnings("ignore", category=FutureWarning)
    logger.info("Starting Auto-Start Camera Detection...")

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.0)
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        return

    win_name = "CAMERA TU DONG DIEM DANH"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    import threading
    import queue
    
    faces = []
    ai_queue = queue.Queue(maxsize=1)
    
    def ai_worker():
        while True:
            try:
                ai_frame = ai_queue.get()
                if ai_frame is None: break
                
                # Chạy AI ở luồng phụ (Màn hình chính sẽ chạy ở 60 FPS mượt mà)
                detected = face_rec.detect_and_extract(ai_frame, max_faces=RecognitionConfig.MAX_FACES)
                tracker.update(detected, attendance, frame=ai_frame, face_rec=face_rec)
                
                nonlocal faces
                faces = detected
            except Exception as e:
                logger.error(f"AI Worker Error: {e}")
            finally:
                ai_queue.task_done()
                
    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    logger.info("Press 'q' to exit camera.")

    try:
        while True:
            if not camera.is_connected:
                if not camera.connect():
                    logger.warning("Camera not connected. Retrying in 5s...")
                    time.sleep(5)
                    continue
            
            success, frame = camera.read_frame()
            if not success or frame is None: 
                continue

            # Đẩy ảnh cho AI phân tích ngầm (Bỏ qua frame nếu AI đang bận)
            if ai_queue.empty():
                try:
                    ai_frame_obj = frame.copy()
                    if CameraConfig.ROI:
                        import numpy as np
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
            
            # CPU lúc này được dồn 100% để NHÌN hình ảnh mượt nhất có thể
            display_frame = face_rec.draw_faces(frame.copy(), faces)
            
            if CameraConfig.ROI:
                import numpy as np
                x1, y1, x2, y2 = CameraConfig.ROI
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(display_frame, "VUNG CHAM CONG", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Tính FPS và hiển thị cùng Ping trên màn hình App Test UI
            current_time = time.time()
            fps = 0
            if hasattr(camera, 'last_frame_time') and camera.last_frame_time:
                fps = 1.0 / (current_time - camera.last_frame_time) if current_time > camera.last_frame_time else 0
            camera.last_frame_time = current_time
            
            ping_str = getattr(camera, 'current_ping', 'N/A')
            
            # HUD Thong ke
            cv2.putText(display_frame, f"FPS: {int(fps)} | Ping: {ping_str} | Press 'q' to stop", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(win_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logger.info("Stopping camera...")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
        logger.info("Camera app shut down.")

if __name__ == "__main__":
    main()
