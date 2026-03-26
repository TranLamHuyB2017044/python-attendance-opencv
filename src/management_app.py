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

# ─── LAZY AI LOADER ──────────────────────────────────────────────────────────
# Management App KHÔNG load AI lúc khởi động.
# AI (FaceRecognition + Camera) chỉ được load KHI USER vào chế độ Enrollment/Edit.
# Điều này giúp tiết kiệm ~300-500MB RAM và ~20-30% CPU khi chạy song song với CameraService.
# ─────────────────────────────────────────────────────────────────────────────

_face_rec = None   # Lazy singleton — chỉ tạo khi cần
_camera   = None   # Lazy singleton — chỉ tạo khi cần enrollment qua camera

def _get_face_rec():
    """Lazy-load FaceRecognition. Chỉ gọi khi thực sự cần (enrollment/edit)."""
    global _face_rec
    if _face_rec is None:
        logger.info("[LazyLoad] Đang tải AI models cho chức năng Đăng ký...")
        # Show loading notification to user
        _show_loading_window("Đang tải AI Models cho chức năng Đăng ký...\nVui lòng chờ trong giây lát.")
        from src.recognition.face_recognition import FaceRecognition
        _face_rec = FaceRecognition()
        _close_loading_window()
        logger.success("[LazyLoad] FaceRecognition loaded thành công.")
    return _face_rec

def _get_camera():
    """Lazy-load RTSPCamera. Chỉ gọi khi enroll qua camera."""
    global _camera
    if _camera is None:
        from src.camera.rtsp_camera import RTSPCamera
        _camera = RTSPCamera()
    return _camera

# Loading window helpers
_loading_win = "Loading AI..."

def _show_loading_window(msg: str):
    """Hiện cửa sổ thông báo loading nhỏ."""
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


def main():
    setup_logger()
    logger.info("Starting Management Interface (No-AI mode)...")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Chặn mở nhiều app cùng lúc
    from src.utils.single_instance import force_single_instance
    force_single_instance("ManagementApp")

    # --- 1. HIỆN MÀN HÌNH LOADING NGAY LẬP TỨC ---
    win_loading = "BITTECH AI SYSTEM"
    cv2.namedWindow(win_loading, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_loading, 1280, 720)

    loading_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(loading_frame, (0, 0), (1280, 720), (30, 30, 30), -1)
    cv2.putText(loading_frame, "BITTECH AI SYSTEM", (440, 300),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(loading_frame, "DANG KHOI TAO HE THONG... VUI LONG CHO TRONG GIAY LAT", (320, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Progress bar
    cv2.rectangle(loading_frame, (440, 420), (840, 430), (60, 60, 60), -1)
    cv2.imshow(win_loading, loading_frame)
    cv2.waitKey(1)

    try:
        # ── Chỉ khởi tạo DATABASE, không load AI ──
        logger.info("Step 1: Connecting to Vector Database (Qdrant)...")
        cv2.rectangle(loading_frame, (440, 420), (640, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        attendance = QdrantAttendanceManager()

        logger.info("Step 2: Connecting to MongoDB...")
        cv2.rectangle(loading_frame, (440, 420), (740, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        CameraConfig.load_from_mongodb(mongo_db)

        logger.info("Step 3: Initializing UI...")
        cv2.rectangle(loading_frame, (440, 420), (840, 430), (0, 255, 0), -1)
        cv2.imshow(win_loading, loading_frame)
        cv2.waitKey(1)
        ui = AttendanceUI()

        # Check service status immediately
        status_doc = mongo_db.db.system_status.find_one({
            "type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID
        })
        if status_doc:
            last_seen = status_doc.get("last_seen", 0)
            ui.service_active = (time.time() - last_seen < 15)

        logger.success("Management App ready (AI models NOT loaded — will lazy-load on demand).")

    except Exception as e:
        logger.critical(f"Khoi tao that bai: {e}")
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
        messagebox.showerror("Lỗi Khởi Tạo", f"Không thể khởi động hệ thống:\n{e}")
        root.destroy()
        return

    win_name = "BITTECH AI SYSTEM"
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

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

        display_frame = ui.draw_main_menu()

        # ─── SHM state for STATE_DETECT ───────────────────────────────────────
        _shm_seq    = -1
        _shm_obj    = None
        _last_frame = None

        while True:
            # Check window size
            try:
                _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
                if cur_w <= 0 or cur_h <= 0:
                    cur_w, cur_h = 1280, 720
            except Exception:
                cur_w, cur_h = 1280, 720

            # Periodically check service heartbeat (every 2 seconds)
            if time.time() - last_heartbeat_check > 2:
                status_doc = mongo_db.db.system_status.find_one({
                    "type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID
                })
                if status_doc:
                    last_seen = status_doc.get("last_seen", 0)
                    service_active = (time.time() - last_seen < 15)
                    ui.camera_connected = status_doc.get("camera_connected", False)
                else:
                    service_active = False
                    ui.camera_connected = False
                last_heartbeat_check = time.time()

            # ── STATE: MENU ────────────────────────────────────────────────────
            if ui.current_state == STATE_MENU:
                try:
                    cv2.destroyWindow(win_name)
                except Exception:
                    pass

                ui.show_main_dashboard(
                    mongo_db, attendance=attendance,
                    face_rec=None,    # Management App không truyền face_rec vào dashboard
                    camera=None,
                    service_active=service_active
                )

                if ui.current_state in [STATE_DETECT, STATE_TEST_CAM]:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, 1280, 720)

                main._detect_cb_cleared = False  # Reset flag để lần sau vào DETECT xóa callback
                last_w, last_h = 0, 0
                continue

            # ── STATE: DETECT (Live Monitor via SHM) ──────────────────────────
            elif ui.current_state == STATE_DETECT:
                if not service_active:
                    from src.utils.notification import show_info_message
                    logger.warning("[Monitor] Service inactive (heartbeat lost).")
                    show_info_message(
                        "Thông báo",
                        "Dịch vụ Camera ẩn đã dừng hoặc chưa chạy.\nQuay lại Menu chính."
                    )
                    ui.current_state = STATE_MENU
                    continue


                # Xóa mouse callback MỘT LẦN khi mới vào STATE_DETECT
                # → Ngăn click camera view vô tình trigger các button từ state trước
                if not getattr(main, '_detect_cb_cleared', False):
                    try:
                        cv2.setMouseCallback(win_name, lambda *args: None)
                    except Exception:
                        pass
                    main._detect_cb_cleared = True

                # Kết nối SHM nếu chưa có
                if _shm_obj is None and service_active:
                    try:
                        from multiprocessing import shared_memory
                        # Explicitly specify create=False
                        _shm_obj = shared_memory.SharedMemory(name="bittech_monitor_shm", create=False)
                        logger.success("[Monitor] Kết nối thành công tới Shared Memory của Camera Service.")
                    except Exception as e:
                        _shm_obj = None
                        # Chỉ log lỗi định kỳ để tránh tràn log
                        if int(time.time()) % 10 == 0:
                            logger.debug(f"[Monitor] Chờ Shared Memory... ({e})")

                preview_img = None
                new_frame   = False

                if _shm_obj is not None:
                    try:
                        # Đọc buffer trực tiếp để tối ưu
                        buf = _shm_obj.buf
                        seq1 = int(buf[0])

                        # Protocol: EVEN sequence = frame hợp lệ (ghi xong)
                        if seq1 % 2 == 0 and seq1 != _shm_seq:
                            seq2 = int(buf[0])
                            if seq1 == seq2:
                                if buf[1] == 1:  # RAW
                                    # Sử dụng np.frombuffer trực tiếp trên memoryview slice (nhanh + an toàn)
                                    w = int(np.frombuffer(buf[2:4], dtype=np.uint16)[0])
                                    h = int(np.frombuffer(buf[4:6], dtype=np.uint16)[0])
                                    
                                    if 100 < w < 3000 and 100 < h < 2000:
                                        size = w * h * 3
                                        offset = 10
                                        if offset + size <= len(buf):
                                            raw = np.frombuffer(buf[offset:offset + size], dtype=np.uint8).reshape((h, w, 3))
                                            preview_img = raw.copy()
                                            _shm_seq = seq1
                                            _last_frame = preview_img
                                            new_frame = True
                    except (ValueError, IndexError, OSError) as e:
                        logger.debug(f"[Monitor] SHM Read error: {e}")
                        _shm_obj = None  # Thử kết nối lại ở vòng lặp sau
                    except Exception as e:
                        pass
                    except OSError:
                        # SHM bị đóng từ phía service → thử kết nối lại
                        _shm_obj = None
                    except Exception:
                        pass  # Đọc thất bại tạm thời → giữ frame cũ
                # Không có frame mới → giữ frame cũ, nhường CPU
                if not new_frame:
                    if _last_frame is not None:
                        preview_img = _last_frame
                    cv2.waitKey(33)  # ~30fps ceiling khi chờ SHM cập nhật
                else:
                    cv2.waitKey(1)

                # Dựng khung hiển thị
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)

                if preview_img is not None:
                    p_h, p_w = preview_img.shape[:2]
                    scale    = min(cur_w / p_w, cur_h / p_h)
                    target_w = int(p_w * scale)
                    target_h = int(p_h * scale)

                    if abs(scale - 1.0) < 0.01:
                        # Pixel-perfect: cùng kích thước → không resize, không mất chất lượng
                        resized = preview_img
                    elif scale < 1.0:
                        # Dùng INTER_LINEAR để mượt mà và nhanh hơn (giống main.py)
                        resized = cv2.resize(preview_img, (target_w, target_h),
                                             interpolation=cv2.INTER_LINEAR)
                    else:
                        # Upscale → INTER_LINEAR nhanh và sắc
                        resized = cv2.resize(preview_img, (target_w, target_h),
                                             interpolation=cv2.INTER_LINEAR)

                    y_off = (cur_h - target_h) // 2
                    x_off = (cur_w - target_w) // 2
                    display_frame[y_off:y_off + target_h, x_off:x_off + target_w] = resized
                else:
                    cv2.putText(
                        display_frame, "DANG KET NOI MONITOR...",
                        (cur_w // 2 - 150, cur_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )

                cv2.putText(display_frame, "[M] Thoat", (10, cur_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

            # ── STATE: ENROLL_CAM ─────────────────────────────────────────────
            elif ui.current_state == STATE_ENROLL_CAM:
                face_rec = _get_face_rec()   # Lazy-load lần đầu
                cam      = _get_camera()

                if not cam.is_connected:
                    if not cam.connect():
                        show_error_message("Lỗi kết nối", "Không thể kết nối với camera để thực hiện đăng ký!")
                        ui.current_state = STATE_MENU
                        continue

                enroll_from_camera(cam, face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            # ── STATE: ENROLL_UPLOAD ──────────────────────────────────────────
            elif ui.current_state == STATE_ENROLL_UPLOAD:
                face_rec = _get_face_rec()   # Lazy-load lần đầu
                enroll_by_upload(face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            # ── STATE: EDIT ───────────────────────────────────────────────────
            elif ui.current_state == STATE_EDIT:
                face_rec = _get_face_rec()   # Lazy-load lần đầu
                cam      = _get_camera()
                handle_edit_logic(attendance, face_rec, ui, cam)
                ui.current_state = STATE_MENU
                continue

            # ── STATE: LIST ───────────────────────────────────────────────────
            elif ui.current_state == STATE_LIST:
                target_cid = get_target_company(ui, mongo_db, allow_selection=True)
                if target_cid:
                    mongo_employees  = mongo_db.get_all_employees(company_id=target_cid)
                    qdrant_employees = attendance.get_all_users(company_id=target_cid)
                    all_employees, seen_ids = [], set()
                    for emp in qdrant_employees:
                        u_id_str = str(emp['user_id'])
                        if u_id_str not in seen_ids:
                            all_employees.append({'user_id': u_id_str, 'user_name': emp['user_name'],
                                                  'birthday': emp['birthday'], 'has_face': True})
                            seen_ids.add(u_id_str)
                    for emp in mongo_employees:
                        if str(emp['user_id']) not in seen_ids:
                            all_employees.append({'user_id': emp['user_id'], 'user_name': emp['name'],
                                                  'birthday': emp.get('birthday', 'N/A'), 'has_face': False})
                            seen_ids.add(str(emp['user_id']))
                    ui.show_user_list_ui(all_employees)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_EDIT:
                face_rec = _get_face_rec()
                cam      = _get_camera()
                handle_edit_logic(attendance, face_rec, ui, cam)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HISTORY:
                target_cid = get_target_company(ui, mongo_db, allow_selection=True)
                if target_cid:
                    company_displayName = mongo_db.get_company_name(target_cid)
                    target_date = ui.get_date_form(
                        title=f"Lịch sử [{company_displayName}]",
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
                    login_res = ui.show_login_dialog()
                    if login_res == "EXIT":
                        break
                    if login_res is True:
                        logged_in = True
                        break

                if not logged_in:
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

            # ── STATE: TEST_CAM (xem camera trực tiếp — không cần AI) ─────────
            elif ui.current_state == STATE_TEST_CAM:
                cam = _get_camera()
                cam_ip   = mongo_db.get_setting("camera_ip",   CameraConfig.IP,   username=ui.session_username)
                cam_port = mongo_db.get_setting("camera_port", CameraConfig.PORT, username=ui.session_username)
                cam_user = mongo_db.get_setting("camera_user", CameraConfig.USER, username=ui.session_username)
                cam_pass = mongo_db.get_setting("camera_pass", CameraConfig.PASS, username=ui.session_username)
                new_url  = f"rtsp://{cam_user}:{cam_pass}@{cam_ip}:{cam_port}/ch1/main"
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

                # Inner loop — chỉ hiển thị raw stream, không AI
                while ui.current_state == STATE_TEST_CAM:
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        f = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        cv2.putText(f, "KHONG THE DOC FRAME", (100, 100), 0, 1, (0, 0, 255), 2)
                        cv2.imshow(win_name, f)
                        if cv2.waitKey(1) & 0xFF == ord('m'):
                            break
                        continue

                    # Chỉ hiển thị raw — KHÔNG chạy AI!
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
                    if cv2.waitKey(1) & 0xFF == ord('m'):
                        break

                cam.disconnect()
                ui.current_state = STATE_MENU
                continue

            # ── RENDER FRAME ───────────────────────────────────────────────────
            final_show = display_frame if ui.current_state in [STATE_DETECT, STATE_TEST_CAM] else ui.frame

            # Check if window was closed via 'X' button
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
        # Dọn dẹp camera nếu đã lazy-load
        if _camera is not None:
            try:
                _camera.disconnect()
            except Exception:
                pass
        if _shm_obj is not None:
            try:
                _shm_obj.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
