import cv2
import time
import warnings
from loguru import logger

# 0. Suppress specific warnings from insightface/skimage
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import CameraConfig, InsightFaceConfig
from src.utils.logger import setup_logger
from src.camera.rtsp_camera import RTSPCamera
from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager


def detect_and_recognize(frame, face_rec, attendance):
    """
    Detect faces in a frame and recognize them against the DB.
    Returns the list of faces with names, IDs, and scores.
    """
    faces = face_rec.detect_and_extract(frame)
    for face in faces:
        name, user_id, score = attendance.recognize(face.normed_embedding)
        face.name = name
        face.user_id = user_id
        face.score = score
    return faces


def enroll_from_camera(camera, face_rec, attendance):
    """
    Experimental function to capture 3-5 samples from camera for enrollment.
    """
    logger.info("Starting Camera Enrollment. Look at the camera.")
    user_id = input("Enter unique ID for enrollment: ").strip()
    if not user_id:
        logger.warning("Enrollment cancelled: No user ID provided.")
        return
        
    user_name = input("Enter name for enrollment: ").strip()
    if not user_name:
        logger.warning("Enrollment cancelled: No name provided.")
        return

    samples = []
    logger.info(f"Collecting 3-5 samples for '{user_name}' (ID: {user_id}). Press 's' to capture a sample, 'c' to cancel.")
    
    while len(samples) < 5:
        success, frame = camera.read_frame()
        if not success or frame is None:
            continue
            
        display_frame = frame.copy()
        faces = face_rec.detect_and_extract(frame)
        
        # Visualize detection for user guidance
        if faces:
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            bbox = faces[0].bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            cv2.putText(display_frame, f"Sample {len(samples)}/5. Press 's' to save", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No face detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enrollment Mode", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if faces:
                samples.append(faces[0].normed_embedding)
                logger.info(f"Captured sample {len(samples)}/5")
                if len(samples) >= 3:
                    logger.info("Minimum samples reached. Press 's' for more (up to 5) or 'f' to finish.")
            else:
                logger.warning("No face detected to capture.")
        
        elif key == ord('f'):
            if len(samples) >= 3:
                break
            else:
                logger.warning(f"Need at least 3 samples (currently {len(samples)}).")
                
        elif key == ord('c'):
            logger.warning("Enrollment aborted by user.")
            cv2.destroyWindow("Enrollment Mode")
            return

    cv2.destroyWindow("Enrollment Mode")
    
    if len(samples) >= 3:
        success = attendance.upsert_user(user_name, user_id, samples)
        if success:
            logger.success(f"User '{user_name}' (ID: {user_id}) successfully enrolled with {len(samples)} samples.")
        else:
            logger.error(f"Failed to enroll user '{user_name}' to Qdrant.")
    else:
        logger.warning("Not enough samples captured. Enrollment failed.")


def main():
    # 1. Setup Logging
    setup_logger()
    logger.info("Starting Face Attendance System...")

    # 2. Initialize Modules
    try:
        # Initialize camera
        camera = RTSPCamera()
        
        # Initialize face recognition (InsightFace)
        face_rec = FaceRecognition()
        
        # Initialize Qdrant manager
        attendance = QdrantAttendanceManager()
        
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 3. Connect to Camera
    if not camera.connect():
        logger.error("Could not connect to RTSP stream. Please check .env config.")
        return

    logger.info("System Ready.")
    logger.info("Commands: 'q': Quit | 'e': Enroll User | 'd': Detect Mode")

    # Metrics
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    processing_time = 0
    
    last_faces = []

    try:
        while True:
            # 4. Read Frame
            success, frame = camera.read_frame()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # 5. Process Detection (Always run for demo, or add toggle)
            ai_start_tick = time.time()
            faces = detect_and_recognize(frame, face_rec, attendance)
            last_faces = faces
            processing_time = (time.time() - ai_start_tick) * 1000

            # 6. Visualization
            display_frame = face_rec.draw_faces(frame, last_faces)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            # Dashboard
            cv2.putText(display_frame, f"UI FPS: {fps} | AI: {processing_time:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"MODEL: {InsightFaceConfig.MODEL_NAME} | DB: QDRANT", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Face Attendance System", display_frame)

            # 7. Interaction
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                enroll_from_camera(camera, face_rec, attendance)

    except KeyboardInterrupt:
        logger.info("System interrupted.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()

