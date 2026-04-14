import cv2
import time
import warnings
import numpy as np
import sys
import os
import threading
import urllib.request
import json
import datetime
from loguru import logger

# --- CẤU HÌNH ĐƯỜNG DẪN CHO PYINSTALLER ---
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    exe_dir = os.path.dirname(sys.executable)
    if exe_dir not in sys.path:
        sys.path.insert(0, exe_dir)
else:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)
# ------------------------------------------

from src.utils.logger import setup_logger
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.attendance.mongodb_mgr import mongo_db
from src.ui.app_ui import (
    AttendanceUI, STATE_MENU, STATE_DETECT, STATE_ENROLL_CAM,
    STATE_ENROLL_UPLOAD, STATE_EDIT, STATE_LIST, STATE_HISTORY,
    STATE_HKB_LIST, STATE_COMPANY, STATE_CLOUD_USER, STATE_LOGOUT, STATE_SETTINGS,
    STATE_TEST_CAM, STATE_EXIT
)
from src.main import enroll_from_camera, enroll_by_upload, get_target_company, handle_edit_logic
from src.config import DATA_DIR, MongoDbConfig, CameraConfig, RecognitionConfig

from src.utils.notification import show_error_message, send_notification
from src.utils.log_panel_renderer import draw_log_panel
from src.utils.webhook_log_bus import start_polling

# ══════════════════════════════════════════════════════════════════════════════
#  LOG PANEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
_LOG_PANEL_RATIO = 0.25          # 25% chiều rộng màn hình

# ── Status colors / short labels ─────────────────────────────────────────────
_S_COLOR = {
    "IN":       (0,  210,  80),
    "OUT":      (50,  90, 240),
    "COOLDOWN": (0,  190, 240),
    "SPOOF":    (0,   50, 230),
    "unknown":  (70,  70,  70),
    "DETECTED": (160, 110,  0),
}
_S_LABEL = {
    "IN": "IN", "OUT": "OUT", "COOLDOWN": "CD",
    "SPOOF": "SP", "unknown": "??", "DETECTED": "DT",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LAZY AI LOADER
# ══════════════════════════════════════════════════════════════════════════════
_face_rec = None
_camera   = None

def _get_face_rec():
    global _face_rec
    if _face_rec is None:
        logger.info("[LazyLoad] Đang tải AI models...")
        _show_loading_window("Đang tải AI Models...\nVui lòng chờ trong giây lát.")
        from src.recognition.face_recognition import FaceRecognition
        _face_rec = FaceRecognition()
        _close_loading_window()
        logger.success("[LazyLoad] FaceRecognition loaded.")
    return _face_rec

def _get_camera():
    global _camera
    if _camera is None:
        from src.camera.rtsp_camera import RTSPCamera
        _camera = RTSPCamera()
    return _camera

_loading_win = "Loading AI..."

def _show_loading_window(msg: str):
    try:
        f = np.zeros((120, 600, 3), dtype=np.uint8)
        cv2.rectangle(f, (0, 0), (600, 120), (30, 30, 30), -1)
        cv2.putText(f, msg.split('\n')[0], (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        if '\n' in msg:
            cv2.putText(f, msg.split('\n')[1], (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.namedWindow(_loading_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_loading_win, 600, 120)
        cv2.imshow(_loading_win, f)
        cv2.waitKey(1)
    except Exception:
        pass

def _close_loading_window():
    try:
        cv2.destroyWindow(_loading_win)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    setup_logger()
    logger.info("Starting Management Interface (No-AI mode)...")
    warnings.filterwarnings("ignore", category=FutureWarning)

    from src.utils.single_instance import force_single_instance
    force_single_instance("ManagementApp")

    # ── Start webhook log panel threads ──────────────────────────────────────
    start_polling()

    # ── Loading screen ────────────────────────────────────────────────────────
    WIN_W, WIN_H = 1600, 900

    win_loading = "BITTECH AI"
    cv2.namedWindow(win_loading, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_loading, WIN_W, WIN_H)

    lf = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    cv2.rectangle(lf, (0, 0), (WIN_W, WIN_H), (28, 28, 28), -1)
    cx = WIN_W // 2
    cv2.putText(lf, "BITTECH AI", (cx - 130, WIN_H // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(lf, "DANG KHOI TAO HE THONG... VUI LONG CHO TRONG GIAY LAT",
                (cx - 300, WIN_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    bar_x0, bar_x1_full = cx - 200, cx + 200
    cv2.rectangle(lf, (bar_x0, WIN_H // 2 + 50), (bar_x1_full, WIN_H // 2 + 62), (55, 55, 55), -1)
    cv2.imshow(win_loading, lf)
    cv2.waitKey(1)

    try:
        logger.info("Step 1: Connecting to Qdrant...")
        cv2.rectangle(lf, (bar_x0, WIN_H // 2 + 50),
                      (bar_x0 + (bar_x1_full - bar_x0) // 3, WIN_H // 2 + 62), (0, 200, 80), -1)
        cv2.imshow(win_loading, lf); cv2.waitKey(1)
        attendance = QdrantAttendanceManager()

        logger.info("Step 2: Connecting to MongoDB...")
        cv2.rectangle(lf, (bar_x0, WIN_H // 2 + 50),
                      (bar_x0 + (bar_x1_full - bar_x0) * 2 // 3, WIN_H // 2 + 62), (0, 200, 80), -1)
        cv2.imshow(win_loading, lf); cv2.waitKey(1)
        CameraConfig.load_from_mongodb(mongo_db)

        logger.info("Step 3: Initializing UI...")
        cv2.rectangle(lf, (bar_x0, WIN_H // 2 + 50),
                      (bar_x1_full, WIN_H // 2 + 62), (0, 200, 80), -1)
        cv2.imshow(win_loading, lf); cv2.waitKey(1)
        ui = AttendanceUI(is_manager_app=True)

        status_doc = mongo_db.db.system_status.find_one({
            "type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID
        })
        if status_doc:
            ui.service_active = (time.time() - status_doc.get("last_seen", 0) < 15)

        logger.success("Management App ready.")

    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
        messagebox.showerror("Lỗi Khởi Tạo", f"Không thể khởi động hệ thống:\n{e}")
        root.destroy()
        return

    win_name = "BITTECH AI"
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Login
    while True:
        login_res = ui.show_login_dialog()
        if login_res == "EXIT":
            logger.info("User thoát tại màn hình đăng nhập.")
            return
        if login_res is True:
            break
        logger.warning("Cửa sổ đăng nhập bị đóng.")

    try:
        service_active     = False
        last_hb_check      = 0
        last_w, last_h     = 0, 0
        _shm_seq           = -1
        _shm_obj           = None
        _last_frame        = None
        _detect_entered    = False   # flag: first time entering STATE_DETECT

        display_frame = ui.draw_main_menu()

        while True:
            # ── Window size ─────────────────────────────────────────────────
            try:
                _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
                if cur_w <= 0 or cur_h <= 0:
                    cur_w, cur_h = WIN_W, WIN_H
            except Exception:
                cur_w, cur_h = WIN_W, WIN_H

            # ── Heartbeat check ─────────────────────────────────────────────
            if time.time() - last_hb_check > 2:
                status_doc = mongo_db.db.system_status.find_one({
                    "type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID
                })
                if status_doc:
                    service_active      = (time.time() - status_doc.get("last_seen", 0) < 15)
                    ui.camera_connected = status_doc.get("camera_connected", False)
                else:
                    service_active      = False
                    ui.camera_connected = False
                last_hb_check = time.time()

            # ══ STATE: MENU ══════════════════════════════════════════════════
            if ui.current_state == STATE_MENU:
                try:
                    cv2.destroyWindow(win_name)
                except Exception:
                    pass

                ui.show_main_dashboard(
                    mongo_db, attendance=attendance,
                    face_rec=None, camera=None,
                    service_active=service_active
                )

                if last_w != WIN_W or last_h != WIN_H:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    init_w = WIN_W if ui.current_state == STATE_DETECT else 1280
                    init_h = WIN_H if ui.current_state == STATE_DETECT else 720
                    cv2.resizeWindow(win_name, init_w, init_h)

                main._detect_cb_cleared = False
                _detect_entered = False
                last_w, last_h  = 0, 0
                continue

            # ══ STATE: DETECT (Live Monitor via Shared Memory) ═══════════════
            elif ui.current_state == STATE_DETECT:
                if not service_active:
                    from src.utils.notification import show_info_message
                    show_info_message(
                        "Thông báo",
                        "Dịch vụ Camera ẩn đã dừng hoặc chưa chạy.\nQuay lại Menu chính."
                    )
                    ui.current_state = STATE_MENU
                    continue

                # Resize window once when first entering STATE_DETECT
                if not _detect_entered:
                    cv2.resizeWindow(win_name, WIN_W, WIN_H)
                    _detect_entered = True

                # Layout:  75% camera | 25% log panel
                panel_w    = max(220, int(cur_w * _LOG_PANEL_RATIO))
                cam_area_w = cur_w - panel_w

                # Clear mouse callback once
                if not getattr(main, '_detect_cb_cleared', False):
                    try:
                        cv2.setMouseCallback(win_name, lambda *args: None)
                    except Exception:
                        pass
                    main._detect_cb_cleared = True

                # Connect to SHM
                if _shm_obj is None and service_active:
                    try:
                        from multiprocessing import shared_memory
                        _shm_obj = shared_memory.SharedMemory(
                            name="bittech_monitor_shm", create=False)
                        logger.success("[Monitor] Shared Memory connected.")
                    except Exception as e:
                        _shm_obj = None
                        if int(time.time()) % 10 == 0:
                            logger.debug(f"[Monitor] Waiting SHM... ({e})")

                preview_img = None
                new_frame   = False

                if _shm_obj is not None:
                    try:
                        buf  = _shm_obj.buf
                        seq1 = int(buf[0])
                        if seq1 % 2 == 0 and seq1 != _shm_seq:
                            seq2 = int(buf[0])
                            if seq1 == seq2 and buf[1] == 1:
                                w = int(np.frombuffer(buf[2:4], dtype=np.uint16)[0])
                                h = int(np.frombuffer(buf[4:6], dtype=np.uint16)[0])
                                if 100 < w < 3000 and 100 < h < 2000:
                                    size = w * h * 3
                                    if 10 + size <= len(buf):
                                        raw = np.frombuffer(
                                            buf[10:10 + size], dtype=np.uint8
                                        ).reshape((h, w, 3))
                                        preview_img = raw.copy()
                                        _shm_seq    = seq1
                                        _last_frame = preview_img
                                        new_frame   = True
                    except (ValueError, IndexError, OSError):
                        _shm_obj = None
                    except Exception:
                        pass

                if not new_frame:
                    preview_img = _last_frame
                    cv2.waitKey(33)
                else:
                    cv2.waitKey(1)

                # ── Compose canvas ───────────────────────────────────────────
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)

                if preview_img is not None:
                    p_h, p_w = preview_img.shape[:2]
                    scale    = min(cam_area_w / p_w, cur_h / p_h)
                    t_w      = int(p_w * scale)
                    t_h      = int(p_h * scale)
                    resized  = cv2.resize(preview_img, (t_w, t_h),
                                          interpolation=cv2.INTER_LINEAR)
                    yo = (cur_h - t_h) // 2
                    xo = (cam_area_w - t_w) // 2
                    display_frame[yo:yo + t_h, xo:xo + t_w] = resized
                else:
                    cv2.putText(
                        display_frame, "DANG KET NOI MONITOR...",
                        (cam_area_w // 2 - 150, cur_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )

                cv2.putText(display_frame, "[M] Thoat / [Q] Menu",
                            (10, cur_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1)

                # ── Log panel ────────────────────────────────────────────────────────
                if panel_w > 0:
                    draw_log_panel(display_frame, cam_area_w, panel_w, cur_h)

            # ══ STATE: ENROLL_CAM ════════════════════════════════════════════
            elif ui.current_state == STATE_ENROLL_CAM:
                face_rec = _get_face_rec()
                cam      = _get_camera()
                if not cam.is_connected:
                    if not cam.connect():
                        show_error_message("Lỗi kết nối",
                                           "Không thể kết nối camera để đăng ký!")
                        ui.current_state = STATE_MENU
                        continue
                enroll_from_camera(cam, face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            # ══ STATE: ENROLL_UPLOAD ═════════════════════════════════════════
            elif ui.current_state == STATE_ENROLL_UPLOAD:
                face_rec = _get_face_rec()
                enroll_by_upload(face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            # ══ STATE: EDIT ══════════════════════════════════════════════════
            elif ui.current_state == STATE_EDIT:
                face_rec = _get_face_rec()
                cam      = _get_camera()
                handle_edit_logic(attendance, face_rec, ui, cam)
                ui.current_state = STATE_MENU
                continue

            # ══ STATE: LIST ══════════════════════════════════════════════════
            elif ui.current_state == STATE_LIST:
                target_cid = get_target_company(ui, mongo_db, allow_selection=True)
                if target_cid:
                    mongo_employees  = mongo_db.get_all_employees(company_id=target_cid)
                    qdrant_employees = attendance.get_all_users(company_id=target_cid)
                    all_employees, seen_ids = [], set()
                    for emp in qdrant_employees:
                        u = str(emp['user_id'])
                        if u not in seen_ids:
                            all_employees.append({'user_id': u, 'user_name': emp['user_name'],
                                                  'birthday': emp['birthday'], 'has_face': True})
                            seen_ids.add(u)
                    for emp in mongo_employees:
                        if str(emp['user_id']) not in seen_ids:
                            all_employees.append({'user_id': emp['user_id'], 'user_name': emp['name'],
                                                  'birthday': emp.get('birthday', 'N/A'), 'has_face': False})
                            seen_ids.add(str(emp['user_id']))
                    ui.show_user_list_ui(all_employees, attendance_manager=attendance)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HISTORY:
                target_cid = get_target_company(ui, mongo_db, allow_selection=True)
                if target_cid:
                    company_name = mongo_db.get_company_name(target_cid)
                    target_date  = ui.get_date_form(
                        title=f"Lịch sử [{company_name}]",
                        ok_button_text="LẤY DỮ LIỆU"
                    )
                    if target_date:
                        logs = mongo_db.get_logs(company_id=target_cid, date=target_date)
                        ui.show_attendance_logs_ui(
                            logs, title=f"Lịch sử - {target_date}",
                            session_role=ui.session_role,
                            session_username=ui.session_username
                        )
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
                ui.session_role       = None
                ui.session_company_id = None
                logged_in = False
                while True:
                    r = ui.show_login_dialog()
                    if r == "EXIT": break
                    if r is True:
                        logged_in = True; break
                if not logged_in:
                    break
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_CLOUD_USER:
                ui.show_user_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_EXIT:
                logger.info("Thoát theo yêu cầu người dùng.")
                break

            elif ui.current_state == STATE_COMPANY:
                ui.show_company_management_ui(mongo_db)
                ui.current_state = STATE_MENU
                continue

            # ══ STATE: TEST_CAM ══════════════════════════════════════════════
            elif ui.current_state == STATE_TEST_CAM:
                cam      = _get_camera()
                cam_ip   = mongo_db.get_setting("camera_ip",   CameraConfig.IP,   username=ui.session_username)
                cam_port = mongo_db.get_setting("camera_port", CameraConfig.PORT, username=ui.session_username)
                cam_user = mongo_db.get_setting("camera_user", CameraConfig.USER, username=ui.session_username)
                cam_pass = mongo_db.get_setting("camera_pass", CameraConfig.PASS, username=ui.session_username)

                new_url = f"rtsp://{cam_user}:{cam_pass}@{cam_ip}:{cam_port}/ch1/main"
                env_url = os.getenv("RTSP_URL")
                if env_url and str(cam_ip) in env_url:
                    new_url = env_url
                if cam_ip.isdigit():
                    new_url = cam_ip

                if str(cam.camera_source) != str(new_url):
                    cam.disconnect()
                    from src.camera.rtsp_camera import RTSPCamera
                    global _camera
                    _camera = RTSPCamera(rtsp_url=str(new_url))
                    cam     = _camera

                if not cam.is_connected:
                    if not cam.connect():
                        show_error_message("Lỗi", f"Không thể kết nối camera tại {cam_ip}!")
                        ui.current_state = STATE_MENU
                        continue

                while ui.current_state == STATE_TEST_CAM:
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        f = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        cv2.putText(f, "KHONG THE DOC FRAME", (100, 100), 0, 1, (0, 0, 255), 2)
                        cv2.imshow(win_name, f)
                        if cv2.waitKey(1) & 0xFF == ord('m'): break
                        continue
                    display = frame.copy()
                    if CameraConfig.ROI:
                        x1, y1, x2, y2 = CameraConfig.ROI
                        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 120, 0), 2)
                        cv2.putText(display, "VUNG CHAM CONG", (x1 + 10, y1 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)
                    cv2.putText(display, "CHEDO TEST CAMERA (KHONG AI)", (10, cur_h - 50),
                                0, 0.7, (0, 165, 255), 2)
                    cv2.putText(display, "[M] Quay ve Menu", (10, cur_h - 20),
                                0, 0.6, (255, 255, 255), 1)
                    cv2.imshow(win_name, display)
                    if cv2.waitKey(1) & 0xFF == ord('m'): break

                cam.disconnect()
                ui.current_state = STATE_MENU
                continue

            # ══ RENDER (DETECT only — other states render their own windows) ═
            final_show = (display_frame
                          if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]
                          else ui.frame)

            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]:
                        ui.current_state = STATE_MENU
                        continue
            except Exception:
                pass

            cv2.imshow(win_name, final_show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]:
                    ui.current_state = STATE_MENU
                else:
                    break
            elif key == ord('m'):
                ui.current_state = STATE_MENU

    finally:
        if _camera is not None:
            try: _camera.disconnect()
            except Exception: pass
        if _shm_obj is not None:
            try: _shm_obj.close()
            except Exception: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
