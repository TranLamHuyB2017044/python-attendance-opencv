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
from src.attendance.mongodb_mgr import mongo_db
from src.recognition.tracker import FaceTracker
from src.config import MongoDbConfig
from src.ui.app_ui import AttendanceUI, STATE_MENU, STATE_DETECT, STATE_ENROLL_CAM, STATE_ENROLL_UPLOAD, STATE_EDIT, STATE_LIST, STATE_HISTORY, STATE_HKB_LIST, STATE_COMPANY, STATE_CLOUD_USER, STATE_LOGOUT, STATE_SETTINGS


def enroll_from_camera(camera, face_rec, attendance, ui):
    """
    Experimental function to capture 3-5 samples from camera for enrollment.
    """
    logger.info("Bat dau dang ky qua Camera. Vui long nhin vao camera.")
    
    # Mở form nhập liệu UI (không cần nút upload)
    user_info = AttendanceUI.get_user_form(include_upload=False, session_role=ui.session_role, mongo_db=mongo_db)
    if not user_info:
        logger.warning("Enrollment cancelled: No user information provided.")
        return
    
    user_id, user_name, birthday, _, selected_cid = user_info

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
            cv2.destroyWindow("Che do Dang ky")
            return

    target_company = MongoDbConfig.COMPANY_ID
    # Use selected company if admin, otherwise session company
    if str(ui.session_role).lower() == 'admin' and selected_cid:
        target_company = selected_cid
    elif ui.session_company_id:
        target_company = ui.session_company_id

    if len(samples) >= 3:
        attendance.upsert_user(user_name, user_id, birthday, samples, company_id=target_company)
        mongo_db.save_employee(user_id, user_name, birthday, target_company)
        logger.success(f"Da dang ky: {user_name} (ID: {user_id}) cho cong ty: {target_company}")
        
        # Show success message
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Thành công", f"Đã đăng ký thành công nhân viên: {user_name} (ID: {user_id})")
        root.destroy()
    
    cv2.destroyWindow("Che do Dang ky")


def enroll_by_upload(face_rec, attendance, ui):
    """
    Enroll users by uploading images from disk.
    """
    # 1. Mở form nhập liệu UI TRƯỚC (có nút chọn ảnh bên trong)
    user_info = AttendanceUI.get_user_form(include_upload=True, session_role=ui.session_role, mongo_db=mongo_db)
    if not user_info:
        logger.warning("Enrollment cancelled: No user information provided.")
        return
        
    u_id, u_name, u_bday, file_paths, selected_cid = user_info
    
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
        target_company = MongoDbConfig.COMPANY_ID
        if str(ui.session_role).lower() == 'admin' and selected_cid:
            target_company = selected_cid
        elif ui.session_company_id:
            target_company = ui.session_company_id
            
        attendance.upsert_user(u_name, u_id, u_bday, samples, company_id=target_company)
        mongo_db.save_employee(u_id, u_name, u_bday, target_company)
        logger.success(f"Enrolled {u_name} via upload for company: {target_company}")
        
        # Show success message
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Thành công", f"Đã đăng ký (Upload) thành công nhân viên: {u_name} (ID: {u_id})")
        root.destroy()


def main():
    setup_logger()
    logger.info("Initializing Face Attendance System...")

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        tracker = FaceTracker(threshold_seconds=2.0)
        ui = AttendanceUI()
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        return

    win_name = "He thong Diem danh Khuon mat"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720) # Default larger size
    
    is_fullscreen = False
    
    fps_start_time = time.time()
    fps_counter, fps = 0, 0

    display_frame = ui.draw_main_menu()
    cv2.imshow(win_name, display_frame)
    if not ui.show_login_dialog():
        logger.warning("Truy cap bi tu choi hoặc ứng dụng bị đóng.")
        return

    try:
        while True:
            # Get actual window size for responsive drawing
            _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
            if cur_w <= 0 or cur_h <= 0:
                cur_w, cur_h = 1280, 720
            
            if ui.current_state == STATE_MENU:
                display_frame = ui.draw_main_menu(w=cur_w, h=cur_h)
                cv2.setMouseCallback(win_name, ui.handle_menu_click, param=(cur_w, cur_h))
                
            elif ui.current_state == STATE_DETECT:
                cv2.setMouseCallback(win_name, lambda *args: None)
                if not camera.is_connected:
                    if not camera.connect():
                        ui.current_state = STATE_MENU
                        continue
                
                success, frame = camera.read_frame()
                if not success: continue

                from src.config import RecognitionConfig
                faces = face_rec.detect_and_extract(frame, max_faces=RecognitionConfig.MAX_FACES)
                tracker.update(faces, attendance, frame)
                display_frame = face_rec.draw_faces(frame, faces)
                
                fps_counter += 1
                if time.time() - fps_start_time > 1.0:
                    fps, fps_counter = fps_counter, 0
                    fps_start_time = time.time()
                
                display_frame = ui.draw_status_bar(display_frame, fps, 0) # Time could be added later

            elif ui.current_state == STATE_ENROLL_CAM:
                if not camera.is_connected: camera.connect()
                
                # Biến cờ để báo hiệu quay lại menu
                enroll_from_camera(camera, face_rec, attendance, ui)
                
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_ENROLL_UPLOAD:
                enroll_by_upload(face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_EDIT:
                # 1. Determine target company
                target_company = ui.session_company_id
                
                # If admin, pick company first
                if str(ui.session_role).lower() == "admin":
                    companies = mongo_db.get_all_companies()
                    if companies:
                        picked = ui.pick_company_ui(companies)
                        if picked:
                            target_company = picked
                        else:
                            ui.current_state = STATE_MENU
                            continue
                    else:
                        # No companies created yet, continue with session_company_id (likely 'admin')
                        pass
                
                # 2. Pick User from that company
                users = attendance.get_all_users(company_id=target_company)
                u_id = ui.pick_user_ui(users)
                
                if u_id:
                    # 3. Get user info and show edit form
                    user_info = attendance.get_user_info(u_id)
                    if user_info:
                        edit_res = AttendanceUI.get_edit_user_form(
                            user_info["user_id"], 
                            user_info["user_name"], 
                            user_info["birthday"]
                        )
                        if edit_res:
                            if edit_res["delete"]:
                                attendance.delete_user(u_id)
                                logger.success(f"Da xoa nhan vien ID: {u_id}")
                            else:
                                attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                                logger.success(f"Da cap nhat thong tin nhan vien ID: {u_id}")
                
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LIST:
                # Logic phân quyền xem danh sách
                target_company = ui.session_company_id
                
                # Nếu là admin, cho phép chọn công ty
                if str(ui.session_role).lower() == "admin":
                    companies = mongo_db.get_all_companies()
                    if companies:
                        picked = ui.pick_company_ui(companies)
                        if picked:
                            target_company = picked
                        else:
                            # Nếu có danh sách mà bấm hủy -> Về menu
                            ui.current_state = STATE_MENU
                            continue
                    else:
                        # Chưa có công ty nào -> Mặc định lấy theo session_company_id (admin)
                        pass
                
                users = attendance.get_all_users(company_id=target_company)
                ui.show_user_list_ui(users)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HISTORY:
                # 1. Logic phân quyền xem lịch sử
                target_company = ui.session_company_id
                
                # Nếu là admin, cho phép chọn công ty
                if str(ui.session_role).lower() == "admin":
                    companies = mongo_db.get_all_companies()
                    if companies:
                        picked = ui.pick_company_ui(companies)
                        if picked:
                            target_company = picked
                        else:
                            ui.current_state = STATE_MENU
                            continue
                
                # 2. Chọn ngày cần xem
                target_date = ui.get_date_form(title=f"Lịch sử [{target_company}]")
                if not target_date:
                    ui.current_state = STATE_MENU
                    continue

                # 3. Filter history by company and date
                logs = mongo_db.get_logs(company_id=target_company, date=target_date)
                
                # Format for display
                display_logs = []
                for l in logs:
                    display_logs.append((
                        str(l["_id"]), l["user_id"], l["user_name"], 
                        l["timestamp"], l["date"], l["status"], l.get("image_webp")
                    ))
                
                ui.show_attendance_logs_ui(display_logs, title=f"Lịch sử ngày {target_date}")
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HKB_LIST:
                ui.show_hkb_connections_ui()
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_COMPANY:
                ui.show_company_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_CLOUD_USER:
                ui.show_user_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_SETTINGS:
                ui.show_system_settings_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LOGOUT:
                logger.info("Logging out...")
                if not ui.show_login_dialog():
                    logger.warning("Logout/Login cancelled. Exiting.")
                    break
                ui.current_state = STATE_MENU
                continue

            cv2.imshow(win_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'):
                ui.current_state = STATE_MENU
                camera.disconnect()
            elif key == ord('f'): # F key to toggle full screen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
