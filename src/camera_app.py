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
from src.config import RecognitionConfig, CameraConfig  # ← thêm CameraConfig

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
    
    faces = []            # List chia sẻ giữa main thread và ai_worker
    ai_queue = queue.Queue(maxsize=1)
    frame_skip_count = 0
    # AI chỉ xử lý 1 trong 3 frame → ở 15fps = ~5 lần/giây, đủ để phản ứng nhanh
    PROCESS_EVERY_N = 3

    def ai_worker():
        nonlocal frame_skip_count
        while True:
            try:
                ai_frame = ai_queue.get()
                if ai_frame is None: break

                # Frame skip: chỉ thực hiện AI mỗi N frame
                frame_skip_count += 1
                if frame_skip_count % PROCESS_EVERY_N != 0:
                    ai_queue.task_done()
                    continue

                # Heavy AI — chạy trong thread phụ, hoàn toàn khồng block main thread
                detected = face_rec.detect_and_extract(ai_frame, max_faces=RecognitionConfig.MAX_FACES)
                tracker.update(detected, attendance, frame=ai_frame, face_rec=face_rec)

                # Cập nhật faces in-place (đồng bộ an toàn qua Python GIL)
                faces[:] = detected
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

            # Đẩy ảnh cho AI phân tích ngầm (đảm bảo luôn là ảnh mới nhất)
            if not ai_queue.empty():
                try:
                    ai_queue.get_nowait()
                except: pass
                
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
            
            # === HUD nhỏ gọn — không banner, chỉ text góc ===
            import datetime
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            current_time = time.time()

            fps = 0.0
            if hasattr(camera, 'last_frame_time') and camera.last_frame_time:
                dt = current_time - camera.last_frame_time
                fps = 1.0 / dt if dt > 0 else 0.0
            camera.last_frame_time = current_time

            ping_str = getattr(camera, 'current_ping', 'N/A')
            try:
                ping_ms = int(ping_str.replace('ms', '')) if 'ms' in ping_str else -1
            except:
                ping_ms = -1

            fps_color  = (0, 230, 0) if fps  >= 12 else (0, 200, 255) if fps  >= 7 else (0, 60, 255)
            ping_color = (0, 230, 0) if ping_ms < 50 else (0, 200, 255) if ping_ms < 150 else (0, 60, 255)
            if ping_ms < 0: ping_color = (160, 160, 160)

            h_f, w_f = display_frame.shape[:2]

            def put_shadow(img, text, pos, scale, color, thick):
                """Text với shadow đen 1px — dễ đọc trên mọi nền."""
                cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1)
                cv2.putText(img, text, pos,              cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thick)

            # Dòng 1: FPS | Ping  (góc trên-trái)
            hud_line = f"FPS {int(fps)}  |  {ping_str}"
            put_shadow(display_frame, f"FPS {int(fps)}", (10, 26), 0.65, fps_color,  2)
            put_shadow(display_frame, f"| {ping_str}",   (95, 26), 0.65, ping_color, 2)

            # Dòng 2: giờ hiện tại  (góc trên-phải)
            put_shadow(display_frame, now_str, (w_f - 95, 26), 0.55, (200, 200, 200), 1)

            # Dòng 3: số mặt (nhỏ, góc dưới-trái)
            face_count = len(faces)
            if face_count > 0:
                put_shadow(display_frame, f"Faces: {face_count}", (10, h_f - 12), 0.5, (160, 220, 160), 1)
            # === END HUD ===
            
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
