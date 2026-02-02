import cv2
import time
import warnings
import numpy as np
from loguru import logger
import tkinter as tk
from tkinter import filedialog

# 0. Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.utils.logger import setup_logger
from src.camera.rtsp_camera import RTSPCamera
from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.recognition.tracker import FaceTracker
from src.ui.app_ui import AttendanceUI, STATE_MENU, STATE_DETECT, STATE_ENROLL_CAM, STATE_ENROLL_UPLOAD, STATE_EDIT, STATE_LIST


def enroll_from_camera(camera, face_rec, attendance):
    """
    Experimental function to capture 3-5 samples from camera for enrollment.
    """
    logger.info("Bat dau dang ky qua Camera. Vui long nhin vao camera.")
    
    # Mở form nhập liệu UI (không cần nút upload)
    user_info = AttendanceUI.get_user_form(include_upload=False)
    if not user_info:
        logger.warning("Enrollment cancelled: No user information provided.")
        return
    
    user_id, user_name, birthday, _ = user_info

    samples = []
    logger.info(f"Collecting 3-5 samples for '{user_name}' (ID: {user_id}). Press 's' to capture a sample, 'c' to cancel.")
    
    while len(samples) < 5:
        success, frame = camera.read_frame()
        if not success or frame is None:
            continue
            
        display_frame = frame.copy()
        faces = face_rec.detect_and_extract(frame)
        
        if faces:
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            bbox = faces[0].bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            cv2.putText(display_frame, f"Mau {len(samples)}/5. Nhan 's' de luu", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Khong tim thay mat!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Che do Dang ky", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if faces:
                samples.append(faces[0].normed_embedding)
                logger.info(f"Captured sample {len(samples)}/5")
            else:
                logger.warning("No face detected to capture.")
        
        elif key == ord('f'):
            if len(samples) >= 3:
                break
                
        elif key == ord('c'):
            logger.warning("Enrollment aborted.")
            cv2.destroyWindow("Enrollment Mode")
            return

    cv2.destroyWindow("Enrollment Mode")
    
    if len(samples) >= 3:
        attendance.upsert_user(user_name, user_id, birthday, samples)
        logger.success(f"Da dang ky: {user_name} (ID: {user_id})")


def enroll_by_upload(face_rec, attendance):
    """
    Enroll users by uploading images from disk.
    """
    # 1. Mở form nhập liệu UI TRƯỚC (có nút chọn ảnh bên trong)
    user_info = AttendanceUI.get_user_form(include_upload=True)
    if not user_info:
        logger.warning("Enrollment cancelled: No user information provided.")
        return
        
    u_id, u_name, u_bday, file_paths = user_info
    
    samples = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None: continue
        faces = face_rec.detect_and_extract(img)
        if faces:
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            samples.append(faces[0].normed_embedding)
            logger.info(f"Extracted from: {path}")

    if samples:
        attendance.upsert_user(u_name, u_id, u_bday, samples)
        logger.success(f"Enrolled {u_name} via upload.")


def main():
    setup_logger()
    logger.info("Initializing Face Attendance System...")

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.5)
        ui = AttendanceUI()
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        return

    win_name = "He thong Diem danh Khuon mat"
    cv2.namedWindow(win_name)
    
    fps_start_time = time.time()
    fps_counter, fps = 0, 0

    try:
        while True:
            if ui.current_state == STATE_MENU:
                display_frame = ui.draw_main_menu()
                cv2.setMouseCallback(win_name, ui.handle_menu_click, param=(800, 600))
                
            elif ui.current_state == STATE_DETECT:
                cv2.setMouseCallback(win_name, lambda *args: None)
                if not camera.is_connected:
                    if not camera.connect():
                        ui.current_state = STATE_MENU
                        continue
                
                success, frame = camera.read_frame()
                if not success: continue

                faces = face_rec.detect_and_extract(frame)
                tracker.update(faces, attendance, frame)
                display_frame = face_rec.draw_faces(frame, faces)
                
                fps_counter += 1
                if time.time() - fps_start_time > 1.0:
                    fps, fps_counter = fps_counter, 0
                    fps_start_time = time.time()
                
                display_frame = ui.draw_status_bar(display_frame, fps, 0) # Time could be added later

            elif ui.current_state == STATE_ENROLL_CAM:
                if not camera.is_connected: camera.connect()
                cv2.destroyWindow(win_name)
                enroll_from_camera(camera, face_rec, attendance)
                ui.current_state = STATE_MENU
                cv2.namedWindow(win_name)
                continue

            elif ui.current_state == STATE_ENROLL_UPLOAD:
                enroll_by_upload(face_rec, attendance)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_EDIT:
                # 1. Nhập ID cần sửa
                u_id = AttendanceUI.get_id_form()
                if u_id:
                    # 2. Tìm thông tin trong DB
                    user_info = attendance.get_user_info(u_id)
                    if user_info:
                        # 3. Hiện Form chỉnh sửa
                        edit_res = AttendanceUI.get_edit_user_form(
                            user_info["user_id"], 
                            user_info["user_name"], 
                            user_info["birthday"]
                        )
                        if edit_res:
                            if edit_res["delete"]:
                                attendance.delete_user(u_id)
                            else:
                                attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    else:
                        from tkinter import messagebox
                        tk_root = tk.Tk()
                        tk_root.withdraw()
                        messagebox.showerror("Lỗi", f"Không tìm thấy nhân viên có ID: {u_id}")
                        tk_root.destroy()
                
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LIST:
                users = attendance.get_all_users()
                ui.show_user_list_ui(users)
                ui.current_state = STATE_MENU
                continue

            cv2.imshow(win_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'):
                ui.current_state = STATE_MENU
                camera.disconnect()

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
