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
    STATE_HKB_LIST, STATE_COMPANY, STATE_CLOUD_USER, STATE_LOGOUT, STATE_SETTINGS
)
from src.main import enroll_from_camera, enroll_by_upload, get_target_company
from src.config import DATA_DIR, MongoDbConfig

def main():
    setup_logger()
    logger.info("Starting Management Interface...")
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        camera = RTSPCamera()
        ui = AttendanceUI()
    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        return

    win_name = "QUAN LY DIEM DANH AI"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    # Initial Login
    if not ui.show_login_dialog():
        logger.warning("Truy cap bi tu choi.")
        return

    try:
        while True:
            _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
            if cur_w <= 0 or cur_h <= 0: cur_w, cur_h = 1280, 720
            
            if ui.current_state == STATE_MENU:
                display_frame = ui.draw_main_menu(w=cur_w, h=cur_h)
                # Overwrite/Modify the Start button behavior if needed
                cv2.setMouseCallback(win_name, ui.handle_menu_click, param=(cur_w, cur_h))
                
            elif ui.current_state == STATE_DETECT:
                # --- LIVE PREVIEW FROM BACKGROUND SERVICE ---
                preview_path = DATA_DIR / "camera_preview.jpg"
                
                # 1. Check Heartbeat Status from MongoDB
                status_doc = mongo_db.db.system_status.find_one({"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID})
                is_active = False
                if status_doc:
                    last_seen = status_doc.get("last_seen", 0)
                    if time.time() - last_seen < 15: # Active in last 15 seconds
                        is_active = True
                
                # 2. Prepare Display Frame
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                
                if os.path.exists(preview_path) and is_active:
                    preview_img = cv2.imread(str(preview_path))
                    if preview_img is not None:
                        # Resize preview to fit nicely in the management window
                        p_h, p_w = preview_img.shape[:2]
                        # Scale to fit (cur_w - 40) width, keeping aspect ratio
                        scale = (cur_w - 60) / p_w
                        target_w = int(p_w * scale)
                        target_h = int(p_h * scale)
                        
                        # If height is too large, scale by height
                        if target_h > cur_h - 200:
                            scale = (cur_h - 220) / p_h
                            target_w = int(p_w * scale)
                            target_h = int(p_h * scale)
                            
                        preview_img = cv2.resize(preview_img, (target_w, target_h))
                        y_off = 100
                        x_off = (cur_w - target_w) // 2
                        display_frame[y_off:y_off+target_h, x_off:x_off+target_w] = preview_img
                else:
                    msg = "DICH VU CAMERA DANG TAT (SERVICE IS OFF)" if not is_active else "DANG DOI ANH PREVIEW (WAITING FOR PREVIEW...)"
                    cv2.putText(display_frame, msg, (cur_w//2 - 350, cur_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Company: {MongoDbConfig.COMPANY_ID}", (cur_w//2 - 150, cur_h//2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                
                # Status Bar inside the monitor view
                status_color = (0, 255, 0) if is_active else (0, 0, 255)
                cv2.rectangle(display_frame, (0, 0), (cur_w, 80), (30, 30, 30), -1)
                cv2.putText(display_frame, "GIAM SAT DICH VU CAMERA (SERVICE MONITOR)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.circle(display_frame, (cur_w - 50, 45), 10, status_color, -1)
                
                # Instruction
                cv2.putText(display_frame, "[M] Quay ve Menu", (20, cur_h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if cv2.waitKey(1) & 0xFF == ord('m'):
                    ui.current_state = STATE_MENU

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
                users = attendance.get_all_users(company_id=ui.session_company_id if ui.session_role != 'admin' else None)
                ui.draw_user_list(users, w=cur_w, h=cur_h)
                cv2.setMouseCallback(win_name, ui.handle_list_click, param=(cur_w, cur_h))
            
            elif ui.current_state == STATE_HISTORY:
                # Fetch logs for the current day by default
                from src.utils.time_manager import time_mgr
                _, date_str = time_mgr.get_formatted_time()
                logs = mongo_db.get_logs(company_id=ui.session_company_id if ui.session_role != 'admin' else None, date=date_str)
                ui.draw_attendance_history(logs, date_str, w=cur_w, h=cur_h)
                cv2.setMouseCallback(win_name, ui.handle_history_click, param=(cur_w, cur_h))

            elif ui.current_state == STATE_EDIT:
                from src.main import handle_edit_logic
                handle_edit_logic(attendance, face_rec, ui, camera)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HKB_LIST:
                from src.services.hkb_service import hkb_service
                connections = hkb_service.get_connections() or []
                ui.draw_hkb_connections(connections, w=cur_w, h=cur_h)
                cv2.setMouseCallback(win_name, ui.handle_hkb_click, param=(cur_w, cur_h))

            elif ui.current_state == STATE_SETTINGS:
                ui.draw_settings_menu(w=cur_w, h=cur_h)
                cv2.setMouseCallback(win_name, ui.handle_settings_click, param=(cur_w, cur_h))

            elif ui.current_state == STATE_LOGOUT:
                ui.session_role = None
                ui.session_company_id = None
                if not ui.show_login_dialog():
                    break
                ui.current_state = STATE_MENU
                continue
            
            # Final display: If we generated a custom display_frame (like in STATE_DETECT), use it.
            # Otherwise fallback to the standard UI frame.
            final_show = display_frame if ui.current_state == STATE_DETECT else ui.frame
            cv2.imshow(win_name, final_show)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        camera.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
