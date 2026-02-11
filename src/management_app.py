import cv2
import time
import warnings
import numpy as np
import sys
import os
from loguru import logger

# --- CẤU HÌNH ĐƯỜNG DẪN CHO PYINSTALLER ---
# --- FIX FOR WINDOWED MODE (NoneType.write error) ---
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)
# ------------------------------------------

from src.utils.logger import setup_logger
from src.camera.rtsp_camera import RTSPCamera
from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.attendance.mongodb_mgr import mongo_db
from src.ui.app_ui import (
    AttendanceUI, STATE_MENU, STATE_DETECT, STATE_ENROLL_CAM, 
    STATE_ENROLL_UPLOAD, STATE_EDIT, STATE_LIST, STATE_HISTORY, 
    STATE_HKB_LIST, STATE_COMPANY, STATE_CLOUD_USER, STATE_LOGOUT, STATE_SETTINGS,
    STATE_TEST_CAM
)
from src.main import enroll_from_camera, enroll_by_upload, get_target_company
from src.config import DATA_DIR, MongoDbConfig, CameraConfig

from src.utils.notification import show_error_message, send_notification

def main():
    setup_logger()
    logger.info("Starting Management Interface...")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Chặn mở nhiều app cùng lúc
    from src.utils.single_instance import force_single_instance
    force_single_instance("ManagementApp")

    # --- 1. HIỆN MÀN HÌNH LOADING NGAY LẬP TỨC ---
    # Tạo cửa sổ OpenCV và phóng to ngay
    win_loading = "QUAN LY DIEM DANH AI"
    cv2.namedWindow(win_loading, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_loading, 1280, 720)
    
    loading_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(loading_frame, (0, 0), (1280, 720), (30, 30, 30), -1)
    
    # Vẽ logo hoặc text loading chuyên nghiệp
    cv2.putText(loading_frame, "BITTECH AI SYSTEM", (440, 300), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(loading_frame, "DANG KHOI TAO HE THONG... VUI LONG CHO TRONG GIAY LAT", (320, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Progress bar giả lập
    cv2.rectangle(loading_frame, (440, 420), (840, 430), (60, 60, 60), -1)
    cv2.imshow(win_loading, loading_frame)
    cv2.waitKey(1)

    try:
        logger.info("Step 1: Initializing Face Recognition models...")
        cv2.rectangle(loading_frame, (440, 420), (540, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        face_rec = FaceRecognition()
        
        logger.info("Step 2: Connecting to Vector Database (Qdrant)...")
        cv2.rectangle(loading_frame, (440, 420), (640, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        attendance = QdrantAttendanceManager()
        
        logger.info("Step 3: Initializing Camera system...")
        cv2.rectangle(loading_frame, (440, 420), (740, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        camera = RTSPCamera()
        
        from src.recognition.tracker import FaceTracker
        tracker = FaceTracker(threshold_seconds=2.0)
        ui = AttendanceUI()
        
        logger.info("Step 4: Syncing with Cloud Database...")
        cv2.rectangle(loading_frame, (440, 420), (840, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        
        # Check service status immediately
        from src.attendance.mongodb_mgr import mongo_db as m_db
        status_doc = m_db.db.system_status.find_one({"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID})
        if status_doc:
            last_seen = status_doc.get("last_seen", 0)
            ui.service_active = (time.time() - last_seen < 15)
            
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        messagebox.showerror("Lỗi Khởi Tạo", f"Không thể khởi động hệ thống:\n{e}")
        root.destroy()
        return

    win_name = "QUAN LY DIEM DANH AI"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    # Initial Login
    if not ui.show_login_dialog():
        logger.warning("Truy cap bi tu choi.")
        return

    try:
        service_active = False
        last_heartbeat_check = 0
        last_w, last_h = 0, 0
        
        display_frame = ui.draw_main_menu() # Initial frame

        while True:
            _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
            if cur_w <= 0 or cur_h <= 0: cur_w, cur_h = 1280, 720
            
            # Periodically check service heartbeat (every 2 seconds)
            if time.time() - last_heartbeat_check > 2:
                status_doc = mongo_db.db.system_status.find_one({"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID})
                if status_doc:
                    last_seen = status_doc.get("last_seen", 0)
                    service_active = (time.time() - last_seen < 15)
                    ui.camera_connected = status_doc.get("camera_connected", False)
                else:
                    service_active = False
                    ui.camera_connected = False
                last_heartbeat_check = time.time()

            if ui.current_state == STATE_MENU:
                display_frame = ui.draw_main_menu(w=cur_w, h=cur_h, service_active=service_active)
                if cur_w != last_w or cur_h != last_h:
                    cv2.setMouseCallback(win_name, ui.handle_menu_click, param=(cur_w, cur_h))
                    last_w, last_h = cur_w, cur_h
                
            elif ui.current_state == STATE_DETECT:
                # --- CHẾ ĐỘ GIÁM SÁT TRỰC TIẾP (SMOOTH STREAM) ---
                cam_ip = mongo_db.get_setting("camera_ip", CameraConfig.IP, username=ui.session_username)
                cam_port = mongo_db.get_setting("camera_port", CameraConfig.PORT, username=ui.session_username)
                cam_user = mongo_db.get_setting("camera_user", CameraConfig.USER, username=ui.session_username)
                cam_pass = mongo_db.get_setting("camera_pass", CameraConfig.PASS, username=ui.session_username)
                
                new_url = f"rtsp://{cam_user}:{cam_pass}@{cam_ip}:{cam_port}/ch1/main"
                if cam_ip.isdigit(): new_url = int(cam_ip)
                
                if not camera.is_connected or str(camera.camera_source) != str(new_url):
                    camera.disconnect()
                    camera = RTSPCamera(rtsp_url=str(new_url))
                    if not camera.connect():
                        from tkinter import messagebox
                        messagebox.showerror("Lỗi", f"Không thể kết nối camera tại {cam_ip}!")
                        ui.current_state = STATE_MENU
                        continue
                
                mon_tracker = FaceTracker(threshold_seconds=1.0)
                f_count = 0
                mon_faces = []
                
                while ui.current_state == STATE_DETECT:
                    success, frame = camera.read_frame()
                    if not success or frame is None:
                        f = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        cv2.putText(f, "KHONG THE DOC FRAME", (cur_w//2-100, cur_h//2), 0, 0.7, (0,0,255), 2)
                        cv2.imshow(win_name, f)
                        if cv2.waitKey(1) & 0xFF == ord('m'): break
                        continue
                    
                    f_count += 1
                    if f_count % 3 == 0:
                        mon_faces = face_rec.detect_and_extract(frame)
                    
                    mon_tracker.update(mon_faces, None, frame, company_id=ui.session_company_id)
                    display_frame = face_rec.draw_faces(frame.copy(), mon_faces)
                    
                    # Chỉ hiển thị video sạch, không có status bar
                    cv2.imshow(win_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('m'):
                        break
                
                camera.disconnect()
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_ENROLL_CAM:
                if not camera.is_connected:
                    if not camera.connect():
                        from tkinter import messagebox
                        messagebox.showerror("Lỗi", "Không thể kết nối camera để đăng ký!")
                        ui.current_state = STATE_MENU
                        continue
                enroll_from_camera(camera, face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_ENROLL_UPLOAD:
                enroll_by_upload(face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LIST:
                from src.main import get_target_company
                target_cid = get_target_company(ui, mongo_db, allow_selection=True)
                if target_cid:
                    mongo_employees = mongo_db.get_all_employees(company_id=target_cid)
                    qdrant_employees = attendance.get_all_users(company_id=target_cid)
                    all_employees = []
                    seen_ids = set()
                    for emp in qdrant_employees:
                        u_id_str = str(emp['user_id'])
                        if u_id_str not in seen_ids:
                            all_employees.append({'user_id': u_id_str, 'user_name': emp['user_name'], 'birthday': emp['birthday'], 'has_face': True})
                            seen_ids.add(u_id_str)
                    for emp in mongo_employees:
                        if str(emp['user_id']) not in seen_ids:
                            all_employees.append({'user_id': emp['user_id'], 'user_name': emp['name'], 'birthday': emp.get('birthday', 'N/A'), 'has_face': False})
                            seen_ids.add(str(emp['user_id']))
                    ui.show_user_list_ui(all_employees)
                ui.current_state = STATE_MENU
                continue
            
            elif ui.current_state == STATE_EDIT:
                from src.main import handle_edit_logic
                handle_edit_logic(attendance, face_rec, ui, camera)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HISTORY:
                target_cid = get_target_company(ui, mongo_db, allow_selection=True)
                if target_cid:
                    target_date = ui.get_date_form(title=f"Lịch sử [{target_cid}]")
                    if target_date:
                        logs = mongo_db.get_logs(company_id=target_cid, date=target_date)
                        ui.show_attendance_logs_ui(logs, title=f"Lịch sử - {target_date}", session_role=ui.session_role, session_username=ui.session_username)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HKB_LIST:
                ui.show_hkb_connections_ui()
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_SETTINGS:
                ui.show_system_settings_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LOGOUT:
                ui.session_role = None
                ui.session_company_id = None
                if not ui.show_login_dialog(): break
                ui.current_state = STATE_MENU
                continue
            
            final_show = display_frame if ui.current_state in [STATE_DETECT, STATE_MENU, STATE_TEST_CAM] else ui.frame
            cv2.imshow(win_name, final_show)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        camera.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
