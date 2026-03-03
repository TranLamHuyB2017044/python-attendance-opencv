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
    STATE_TEST_CAM, STATE_EXIT
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
    win_loading = "BITTECH AI SYSTEM"
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
        
        from src.attendance.mongodb_mgr import mongo_db as m_db
        CameraConfig.load_from_mongodb(m_db)
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

    win_name = "BITTECH AI SYSTEM"
    # Close loading window before showing login dialog to keep UI clean
    try: cv2.destroyAllWindows()
    except: pass

    # Initial Login Loop
    while True:     
        login_res = ui.show_login_dialog()
        if login_res == "EXIT":
            logger.info("Người dùng chọn thoát tại màn hình đăng nhập.")
            return
        if login_res is True:
            break
        logger.warning("Cửa sổ đăng nhập bị đóng. Vui lòng đăng nhập để tiếp tục.")

    try:
        service_active = False
        last_heartbeat_check = 0
        last_w, last_h = 0, 0
        
        display_frame = ui.draw_main_menu() # Initial frame

        while True:
            # Check window size only if it exists
            try:
                # This call fails if win_name does not exist (e.g., when Dashboard is active)
                # We catch it gracefully to avoid app crash
                _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
                if cur_w <= 0 or cur_h <= 0: cur_w, cur_h = 1280, 720
            except:
                cur_w, cur_h = 1280, 720
            
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
                # hide main opencv window while dashboard is active
                try: cv2.destroyWindow(win_name)
                except: pass
                
                # Show modern dashboard (Blocks until action selected)
                ui.show_main_dashboard(mongo_db, attendance=attendance, face_rec=face_rec, camera=camera, service_active=service_active)
                
                # Only recreate window if we are moving to a state that actually needs it
                # Enrollment (Cam) handles its own window, Detect needs the main win_name
                if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, 1280, 720)
                
                last_w, last_h = 0, 0 # Force resize update next time
                continue
                
            elif ui.current_state == STATE_DETECT:
                if not service_active:
                    from src.utils.notification import show_info_message
                    show_info_message("Thông báo", "Dịch vụ Camera ẩn đã dừng hoặc chưa chạy.\nQuay lại Menu chính.")
                    ui.current_state = STATE_MENU
                    continue

                # --- LIVE PREVIEW FROM SHARED MEMORY ---
                # Khởi tạo một lần cho STATE_DETECT session
                if not hasattr(main, '_shm_seq'):
                    main._shm_seq     = -1      # sequence tracker
                    main._shm_obj     = None    # SHM handle
                    main._last_frame  = None    # cache frame hợp lệ cuối

                # Kết nối SHM nếu chưa có
                if main._shm_obj is None and service_active:
                    try:
                        from multiprocessing import shared_memory
                        main._shm_obj = shared_memory.SharedMemory(name="bittech_monitor_shm")
                    except Exception:
                        main._shm_obj = None

                preview_img = None
                new_frame   = False

                if main._shm_obj is not None:
                    try:
                        seq1 = int(main._shm_obj.buf[0])
                        # Chỉ xử lý khi sequence EVEN (service đã viết xong) VÀ là frame MỚI
                        if seq1 % 2 == 0 and seq1 > 0 and seq1 != main._shm_seq:
                            seq2 = int(main._shm_obj.buf[0])        # double-check không race
                            if seq1 == seq2:
                                fmt = main._shm_obj.buf[1]
                                if fmt == 1:  # RAW pixels
                                    w = int(np.frombuffer(bytes(main._shm_obj.buf[2:4]), dtype=np.uint16)[0])
                                    h = int(np.frombuffer(bytes(main._shm_obj.buf[4:6]), dtype=np.uint16)[0])
                                    if 100 < w < 4000 and 100 < h < 4000:
                                        size = w * h * 3
                                        if 0 < size < 4_800_000:
                                            # Zero-copy view → reshape (không cần bytes() copy)
                                            raw = np.frombuffer(
                                                bytes(main._shm_obj.buf[10:10+size]),
                                                dtype=np.uint8
                                            ).reshape((h, w, 3))
                                            preview_img = raw.copy()    # copy nhỏ để tránh SHM race
                                            main._shm_seq    = seq1     # đánh dấu đã đọc
                                            main._last_frame = preview_img
                                            new_frame = True
                    except Exception:
                        main._shm_obj = None

                # Không có frame mới → giữ frame cũ, sleep ngắn để nhường CPU cho service
                if not new_frame:
                    if main._last_frame is not None:
                        preview_img = main._last_frame
                    cv2.waitKey(16)     # ~60fps ceiling + nhường CPU cho service_main
                else:
                    cv2.waitKey(1)

                # Dựng khung hiển thị
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)

                if preview_img is not None:
                    p_h, p_w = preview_img.shape[:2]
                    scale = min(cur_w / p_w, cur_h / p_h)
                    target_w = int(p_w * scale)
                    target_h = int(p_h * scale)
                    resized   = cv2.resize(preview_img, (target_w, target_h))
                    y_off     = (cur_h - target_h) // 2
                    x_off     = (cur_w - target_w) // 2
                    display_frame[y_off:y_off+target_h, x_off:x_off+target_w] = resized
                else:
                    cv2.putText(display_frame, "DANG KET NOI MONITOR...",
                                (cur_w//2 - 150, cur_h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                cv2.putText(display_frame, "[M] Thoat", (10, cur_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)



            elif ui.current_state == STATE_ENROLL_CAM:
                if not camera.is_connected:
                    if not camera.connect():
                        from src.utils.notification import show_error_message
                        show_error_message("Lỗi kết nối", "Không thể kết nối với camera vật lý để thực hiện đăng ký!")
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
                    company_displayName = mongo_db.get_company_name(target_cid)
                    target_date = ui.get_date_form(title=f"Lịch sử [{company_displayName}]", ok_button_text="LẤY DỮ LIỆU")
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
                
                # Re-login loop
                logged_in = False
                while True:
                    login_res = ui.show_login_dialog()
                    if login_res == "EXIT":
                        break
                    if login_res is True:
                        logged_in = True
                        break
                
                if not logged_in: # User exited via "EXIT"
                    break
                    
                ui.current_state = STATE_MENU
                continue
            
            elif ui.current_state == STATE_CLOUD_USER:
                ui.show_user_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue
            
            elif ui.current_state == STATE_EXIT:
                logger.info("Thoát ứng dụng theo yêu cầu người dùng.")
                break

            elif ui.current_state == STATE_COMPANY:
                ui.show_company_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue
            
            final_show = display_frame if ui.current_state in [STATE_DETECT, STATE_MENU, STATE_TEST_CAM] else ui.frame
            
            # Check if window was closed via 'X' button
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                # If window was closed but we are in a state that needs it, go back to menu
                if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]:
                    ui.current_state = STATE_MENU
                    camera.disconnect()
                    continue

            cv2.imshow(win_name, final_show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                # If in Live Monitor, 'q' goes back to menu, NOT exit app (as per user request: "mới tắt app" usually means from main screen)
                # Wait, user said "nên ấn nút close hoặc q mới tắt app".
                # If they are in Live Monitor, maybe 'q' should exit? No, usually 'q' is back to menu.
                # Let's make 'q' in Live Monitor go back to menu, but if in MENU/DASHBOARD (where OpenCV window is hidden), this waitKey isn't even reached.
                if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]:
                    ui.current_state = STATE_MENU
                    camera.disconnect()
                else:
                    break
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
