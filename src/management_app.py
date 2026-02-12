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
from src.main import enroll_from_camera, enroll_by_upload, get_target_company, handle_edit_logic
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
                if not service_active:
                    from tkinter import messagebox
                    import threading
                    def show_warn():
                        import tkinter as tk
                        msg_root = tk.Tk(); msg_root.withdraw(); msg_root.attributes('-topmost', True)
                        messagebox.showwarning("Dịch Vụ Đang Tắt", "Dịch vụ Camera ẩn chưa chạy.\n\nHướng dẫn:\n1. Vui lòng mở file 'service_main.exe' trước khi xem live.")
                        msg_root.destroy()
                    threading.Thread(target=show_warn, daemon=True).start()
                    ui.current_state = STATE_MENU
                    continue

                # --- LIVE PREVIEW FROM BACKGROUND SERVICE (MJPEG STREAM) ---
                from src.config import DATA_DIR
                preview_path = DATA_DIR / "camera_preview.jpg"
                
                # Prepare Display Frame
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                preview_img = None

                # --- LIVE PREVIEW FROM SHARED MEMORY (ULTRA STABLE) ---
                from src.config import DATA_DIR
                preview_path = DATA_DIR / "camera_preview.jpg"
                
                # Prepare Display Frame
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                preview_img = None

                if service_active:
                    try:
                        from multiprocessing import shared_memory
                        # Connect or Reset SHM
                        if not hasattr(main, 'shm_obj') or main.shm_obj is None:
                            try:
                                main.shm_obj = shared_memory.SharedMemory(name="bittech_monitor_shm")
                            except:
                                main.shm_obj = None
                        
                        if main.shm_obj is not None:
                            # 1. READ SEQUENCE START
                            seq1 = int(main.shm_obj.buf[0])
                            
                            # Valid data only if sequence is EVEN
                            if seq1 % 2 == 0:
                                # 2. Read size
                                size_bytes = main.shm_obj.buf[1:5]
                                size = np.frombuffer(size_bytes, dtype=np.uint32)[0]
                                
                                # Safety check for size
                                if 100 < size < 4.8 * 1024 * 1024:
                                    # 3. Read JPEG data
                                    img_data = bytes(main.shm_obj.buf[5:5+size])
                                    
                                    # 4. READ SEQUENCE END
                                    seq2 = int(main.shm_obj.buf[0])
                                    
                                    # Confirm data was NOT changed during read
                                    if seq1 == seq2:
                                        nparr = np.frombuffer(img_data, dtype=np.uint8)
                                        # Use high-performance imdecode
                                        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                        if decoded is not None:
                                            preview_img = decoded
                                            main.last_valid_frame = preview_img
                    except Exception as e:
                        # Reset SHM object on any error to self-heal
                        main.shm_obj = None

                # --- ULTIMATE FLICKER & FREEZE PREVENTION ---
                if preview_img is None and hasattr(main, 'last_valid_frame'):
                    preview_img = main.last_valid_frame
                
                if preview_img is not None:
                    p_h, p_w = preview_img.shape[:2]
                    # True Full screen scaling (fit to window)
                    scale_w = cur_w / p_w
                    scale_h = cur_h / p_h 
                    scale = min(scale_w, scale_h)
                    
                    target_w = int(p_w * scale)
                    target_h = int(p_h * scale)
                    preview_img = cv2.resize(preview_img, (target_w, target_h))
                    
                    y_off = (cur_h - target_h) // 2
                    x_off = (cur_w - target_w) // 2
                    display_frame[y_off:y_off+target_h, x_off:x_off+target_w] = preview_img
                else:
                    cv2.putText(display_frame, "DANG KET NOI MONITOR...", (cur_w//2 - 150, cur_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Subtle hint at the bottom
                cv2.putText(display_frame, "[M] Thoat", (10, cur_h - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)


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
            
            elif ui.current_state == STATE_CLOUD_USER:
                ui.show_user_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue
            
            elif ui.current_state == STATE_COMPANY:
                ui.show_company_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue
            
            final_show = display_frame if ui.current_state in [STATE_DETECT, STATE_MENU, STATE_TEST_CAM] else ui.frame
            cv2.imshow(win_name, final_show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'):
                ui.current_state = STATE_MENU
                camera.disconnect()
                # Release stream if active
                if hasattr(main, 'stream_cap') and main.stream_cap is None:
                    main.stream_cap = None
                elif hasattr(main, 'stream_cap') and main.stream_cap is not None:
                    main.stream_cap.release()
                    main.stream_cap = None


    finally:
        camera.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
