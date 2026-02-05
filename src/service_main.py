import cv2
import time
import warnings
from loguru import logger
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
            
            # Small sleep to manage CPU usage if camera is too fast
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
