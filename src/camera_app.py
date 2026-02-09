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

            # Process detection
            faces = face_rec.detect_and_extract(frame, max_faces=RecognitionConfig.MAX_FACES)
            
            # Update tracker and log attendance
            tracker.update(faces, attendance, frame)
            
            # Draw results
            display_frame = face_rec.draw_faces(frame, faces)
            
            # Add status info
            cv2.putText(display_frame, f"System Running - Press 'q' to stop", (20, 40),
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
