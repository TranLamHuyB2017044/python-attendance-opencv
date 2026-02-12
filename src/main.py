import cv2
import time
import os
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
from src.config import MongoDbConfig, CameraConfig, DATA_DIR
from src.ui.app_ui import AttendanceUI, STATE_MENU, STATE_DETECT, STATE_ENROLL_CAM, STATE_ENROLL_UPLOAD, STATE_EDIT, STATE_LIST, STATE_HISTORY, STATE_HKB_LIST, STATE_COMPANY, STATE_CLOUD_USER, STATE_LOGOUT, STATE_SETTINGS, STATE_TEST_CAM


def get_target_company(ui, mongo_db, allow_selection=True, parent=None):
    """
    Determine target company based on user role.
    
    Args:
        ui: AttendanceUI instance with session info
        mongo_db: MongoDbManager instance
        allow_selection: If True and user is admin, allow company selection
    
    Returns:
        str: Company ID to use for operations, or None if cancelled
    """
    # Company users can ONLY access their own company
    if str(ui.session_role).lower() == "company":
        cid = ui.session_company_id or ""
        logger.info(f"Company user '{ui.session_username}' accessing their company: '{cid}'")
        return cid
    
    # Admin can select company (if allowed) or use their session company
    if str(ui.session_role).lower() == "admin":
        if allow_selection:
            companies = mongo_db.get_all_companies()
            if companies:
                picked = ui.pick_company_ui(companies, parent=parent)
                if picked:
                    logger.info(f"Admin selected company: {picked}")
                    return picked
                else:
                    logger.warning("Company selection cancelled.")
                    return None
            else:
                logger.warning("No companies found in database.")
                return "" # Fallback
        else:
            cid = ui.session_company_id or ""
            logger.info(f"Admin using session company: '{cid}'")
            return cid
    
    return "" # Default fallback


def enroll_from_camera(camera, face_rec, attendance, ui):
    """
    Enroll a user by capturing face samples from the camera.
    """
    from src.attendance.mongodb_mgr import mongo_db
    from src.config import MongoDbConfig
    
    user_info = ui.get_user_form(
        include_upload=False, 
        session_role=ui.session_role,
        mongo_db=mongo_db
    )
    
    if not user_info:
        logger.warning("Enrollment cancelled: No user information provided.")
        return
    
    user_id, user_name, birthday, _, selected_cid = user_info

    samples = []
    logger.info(f"Collecting 3-5 samples for '{user_name}' (ID: {user_id}). Press 's' to capture, 'f' to finish, 'm' to menu, 'c' to cancel.")
    
    try:
        while len(samples) < 5:
            success, frame = camera.read_frame()
            if not success or frame is None:
                continue
                
            display_frame = frame.copy()
            faces = face_rec.detect_and_extract(frame)
            
            # Get frame dimensions
            h, w = display_frame.shape[:2]
            
            if faces:
                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                bbox = faces[0].bbox.astype(int)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cv2.putText(display_frame, f"Mau {len(samples)}/5", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Khong tim thay mat!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add instruction panel at bottom
            cv2.rectangle(display_frame, (0, h-60), (w, h), (0, 0, 0), -1)
            cv2.putText(display_frame, "[S] Luu mau  [F] Hoan thanh  [M] Menu  [C] Huy", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Che do Dang ky", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if faces:
                    samples.append(faces[0].normed_embedding)
                    logger.info(f"Captured sample {len(samples)}/5")
                else:
                    logger.warning("No face detected to capture.")
            
            elif key == ord('f'):
                if len(samples) >= 1:
                    break
            
            elif key == ord('m'):
                logger.info("Returning to menu...")
                return
                    
            elif key == ord('c'):
                logger.warning("Enrollment cancelled by user.")
                return

        target_company = MongoDbConfig.COMPANY_ID
        # Use selected company if admin, otherwise session company
        if str(ui.session_role).lower() == 'admin' and selected_cid:
            target_company = selected_cid
        elif ui.session_company_id:
            target_company = ui.session_company_id

        if len(samples) >= 1:
            # --- Check for existing user to ask before update ---
            force_upd = False
            existing = mongo_db.employees.find_one({"user_id": str(user_id), "company_id": target_company})
            if existing:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                if messagebox.askyesno("Xác nhận", f"Mã nhân viên '{user_id}' đã tồn tại trong hệ thống.\n\nBạn có muốn CẬP NHẬT dữ liệu mới nhất cho nhân viên này không?"):
                    force_upd = True
                else:
                    logger.info("User cancelled update.")
                    root.destroy()
                    return
                root.destroy()

            # Try to save to MongoDB
            ok, msg = mongo_db.save_employee(user_id, user_name, birthday, target_company, force_update=force_upd)
            if not ok:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                messagebox.showerror("Lỗi đăng ký", f"Không thể lưu nhân viên: {msg}")
                root.destroy()
                return

            ok_qdrant = attendance.upsert_user(user_name, user_id, birthday, samples, clear_old=force_upd, company_id=target_company)
            if ok_qdrant:
                logger.success(f"Da dang ky: {user_name} (ID: {user_id}) cho cong ty: {target_company}")
                
                # Show success message
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                messagebox.showinfo("Thành công", f"Đã đăng ký thành công nhân viên: {user_name} (ID: {user_id})")
                root.destroy()
            else:
                logger.error(f"Failed to save face data to Qdrant for {user_id}")
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                messagebox.showerror("Lỗi", "Đã lưu thông tin nhân viên nhưng thất bại khi đăng ký khuôn mặt.")
                root.destroy()
    
    finally:
        # Always cleanup window
        try:
            cv2.destroyWindow("Che do Dang ky")
        except:
            pass


def enroll_by_upload(face_rec, attendance, ui, parent=None):
    """
    Enroll users by uploading images from disk.
    """
    try:
        # 1. Mở form nhập liệu UI TRƯỚC (có nút chọn ảnh bên trong)
        logger.info("Opening user enrollment form...")
        user_info = AttendanceUI.get_user_form(include_upload=True, session_role=ui.session_role, mongo_db=mongo_db, parent=parent)
        
        if not user_info:
            logger.warning("Enrollment cancelled: No user information provided.")
            return
            
        u_id, u_name, u_bday, file_paths, selected_cid = user_info
        logger.info(f"Processing enrollment for {u_name} (ID: {u_id}) with {len(file_paths)} files.")
        
        samples = []
        for i, path in enumerate(file_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(file_paths)}: {path}")
                img = cv2.imread(path)
                if img is None:
                    logger.error(f"Could not read image: {path}")
                    continue
                    
                faces = face_rec.detect_and_extract(img)
                if faces:
                    # Sort by face size to get the most prominent face
                    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    samples.append(faces[0].normed_embedding)
                    logger.info(f"Successfully extracted face embedding from: {path}")
                else:
                    logger.warning(f"No face detected in image: {path}")
            except Exception as img_err:
                logger.error(f"Error processing image {path}: {img_err}")

        if samples:
            logger.info(f"Face extraction complete. {len(samples)} valid samples found.")
            target_company = MongoDbConfig.COMPANY_ID
            if str(ui.session_role).lower() == 'admin' and selected_cid:
                target_company = selected_cid
            elif ui.session_company_id:
                target_company = ui.session_company_id
                
            # Update databases
            logger.info(f"Saving to Qdrant and MongoDB for company: {target_company}")
            
            # --- Check for existing user to ask before update ---
            force_upd = False
            existing = mongo_db.employees.find_one({"user_id": str(u_id), "company_id": target_company})
            if existing:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                if messagebox.askyesno("Xác nhận", f"Mã nhân viên '{u_id}' đã tồn tại trong hệ thống.\n\nBạn có muốn CẬP NHẬT dữ liệu mới nhất cho nhân viên này không?"):
                    force_upd = True
                else:
                    logger.info("User cancelled update.")
                    root.destroy()
                    return
                root.destroy()

            # Validation via MongoDB
            ok, msg = mongo_db.save_employee(u_id, u_name, u_bday, target_company, force_update=force_upd)
            if not ok:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                messagebox.showerror("Lỗi đăng ký", f"Không thể lưu nhân viên: {msg}")
                root.destroy()
                return

            attendance.upsert_user(u_name, u_id, u_bday, samples, clear_old=force_upd, company_id=target_company)
            logger.success(f"Successfully enrolled {u_name} via upload.")
            
            # Show success message using a robust method
            try:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                messagebox.showinfo("Thành công", f"Đã đăng ký (Upload) thành công nhân viên: {u_name}")
                root.destroy()
            except:
                pass
        else:
            logger.error("No valid face samples were extracted from the provided files.")
            try:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                messagebox.showwarning("Lỗi", "Không tìm thấy khuôn mặt hợp lệ trong các ảnh đã chọn!")
                root.destroy()
            except:
                pass

    except Exception as e:
        logger.error(f"CRITICAL ERROR in enroll_by_upload: {e}")
        import traceback
        logger.error(traceback.format_exc())


def handle_edit_logic(attendance, face_rec, ui, camera, parent=None):
    """
    Handles the sequence for editing a user: Company Selection -> User Picking -> Details Edit -> (Optional) Re-enroll.
    """
    # 1. Determine target company with access control
    target_company = get_target_company(ui, mongo_db, allow_selection=True, parent=parent)
    
    if target_company is None:
        return
    
    # For company users, we also want to see employees from their connected HKB systems
    filter_company = target_company
    if str(ui.session_role).lower() == "company":
        conns = list(mongo_db.auth_services.find({"user_id": ui.session_user_id}))
        if conns:
            filter_company = [target_company] if target_company else []
            for c in conns:
                if c["uuid"] not in filter_company:
                    filter_company.append(c["uuid"])
            logger.info(f"Company user '{ui.session_username}' editing merged list for IDs: {filter_company}")

    # 2. Get merged employee list (MongoDB + Qdrant) for this company ONLY
    mongo_employees = mongo_db.get_all_employees(company_id=filter_company)
    qdrant_employees = attendance.get_all_users(company_id=filter_company)
    
    # Merge employee lists
    all_employees = []
    seen_ids = set()
    
    # First add all from Qdrant (have face data)
    for emp in qdrant_employees:
        u_id_str = str(emp['user_id'])
        if u_id_str not in seen_ids:
            all_employees.append({
                'user_id': u_id_str,
                'user_name': emp['user_name'],
                'birthday': emp['birthday'],
                'has_face': True
            })
            seen_ids.add(u_id_str)
    
    # Then add MongoDB-only employees (no face data yet)
    for emp in mongo_employees:
        # Normalize to string for comparison
        if str(emp['user_id']) not in seen_ids:
            all_employees.append({
                'user_id': emp['user_id'],
                'user_name': emp['name'],
                'birthday': emp.get('birthday', 'N/A'),
                'has_face': False
            })
            seen_ids.add(str(emp['user_id']))
    
    # 3. Pick User from merged list
    u_id = ui.pick_user_ui(all_employees, parent=parent)
    
    if u_id:
        logger.info(f"Selected user_id for edit: {u_id}")
        
        # 4. Get user info - try Qdrant first, then MongoDB
        user_info = attendance.get_user_info(u_id)
        
        if not user_info:
            logger.info(f"User {u_id} not found in Qdrant, checking MongoDB...")
            mongo_emp = next((e for e in mongo_employees if str(e['user_id']) == str(u_id)), None)
            if mongo_emp:
                user_info = {
                    "user_id": mongo_emp['user_id'],
                    "user_name": mongo_emp['name'],
                    "birthday": mongo_emp.get('birthday', 'N/A')
                }
            else:
                logger.warning(f"User {u_id} not found in MongoDB either!")
        
        if user_info:
            edit_res = ui.get_edit_user_form(
                user_info["user_id"], 
                user_info["user_name"], 
                user_info["birthday"],
                session_role=ui.session_role,
                parent=parent
            )
            if edit_res:
                if edit_res["delete"]:
                    # Delete from both Qdrant and MongoDB
                    attendance.delete_user(u_id)
                    mongo_db.employees.delete_one({"user_id": str(u_id), "company_id": target_company})
                    logger.success(f"Da xoa nhan vien ID: {u_id}")
                
                elif edit_res["enroll_camera"]:
                    # Update thong tin trước, sau đó dang ký qua camera
                    attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company, force_update=True)
                    
                    if not camera.is_connected: camera.connect()
                    
                    # Trigger collection
                    samples = []
                    try:
                        while len(samples) < 5:
                            success, frame = camera.read_frame()
                            if not success or frame is None: continue
                            
                            display_frame = frame.copy()
                            faces = face_rec.detect_and_extract(frame)
                            h, w = display_frame.shape[:2]
                            
                            if faces:
                                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                                bbox = faces[0].bbox.astype(int)
                                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                                cv2.putText(display_frame, f"Mau {len(samples)}/5", (10, 30), 0, 0.8, (255, 255, 0), 2)
                            else:
                                cv2.putText(display_frame, "Khong tim thay mat!", (10, 30), 0, 0.8, (0, 0, 255), 2)
                            
                            cv2.rectangle(display_frame, (0, h-60), (w, h), (0, 0, 0), -1)
                            cv2.putText(display_frame, "[S] Luu mau  [F] Hoan thanh  [M] Menu  [C] Huy", (10, h-20), 0, 0.7, (255, 255, 255), 2)
                            cv2.imshow("Che do Dang ky", display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('s'):
                                if faces:
                                    samples.append(faces[0].normed_embedding)
                                    logger.info(f"Captured sample {len(samples)}/5")
                            elif key == ord('f') and len(samples) >= 1: break
                            elif key in [ord('m'), ord('c')]: break
                        
                        if len(samples) >= 1:
                            attendance.upsert_user(edit_res["name"], u_id, edit_res["bday"], samples, clear_old=True, company_id=target_company)
                            from tkinter import messagebox
                            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                            messagebox.showinfo("Thành công", f"Đã đăng ký khuôn mặt cho {edit_res['name']}")
                            root.destroy()
                    finally:
                        try: cv2.destroyWindow("Che do Dang ky")
                        except: pass
                
                elif edit_res["enroll_upload"]:
                    # Update thong tin truoc
                    attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company, force_update=True)
                    
                    from tkinter import filedialog
                    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                    file_paths = filedialog.askopenfilenames(title="Chọn ảnh khuôn mặt", filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")])
                    root.destroy()
                    
                    if file_paths:
                        samples = []
                        for fp in file_paths[:5]:
                            img = cv2.imread(fp)
                            if img is not None:
                                faces = face_rec.detect_and_extract(img)
                                if faces: samples.append(faces[0].normed_embedding)
                        
                        if len(samples) >= 1:
                            attendance.upsert_user(edit_res["name"], u_id, edit_res["bday"], samples, clear_old=True, company_id=target_company)
                            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                            messagebox.showinfo("Thành công", f"Đã cập nhật {len(samples)} ảnh cho {edit_res['name']}")
                            root.destroy()
                else:
                    # Only update info
                    mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company, force_update=True)
                    attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    logger.success(f"Da cap nhat thong tin ID: {u_id}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Lỗi", "Không tìm thấy thông tin nhân viên")


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
        service_active = False
        last_heartbeat_check = 0
        last_w, last_h = 0, 0
        
        while True:
            # Get actual window size for responsive drawing
            _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
            if cur_w <= 0 or cur_h <= 0:
                cur_w, cur_h = 1280, 720
            
            # Periodically check service heartbeat (every 2 seconds)
            if time.time() - last_heartbeat_check > 2:
                from src.attendance.mongodb_mgr import mongo_db as m_db
                status_doc = m_db.db.system_status.find_one({"type": "camera_service", "company_id": MongoDbConfig.COMPANY_ID})
                if status_doc:
                    last_seen = status_doc.get("last_seen", 0)
                    service_active = (time.time() - last_seen < 15)
                else:
                    service_active = False
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
                        msg_root = tk.Tk(); msg_root.withdraw(); msg_root.attributes('-topmost', True)
                        messagebox.showwarning("Dịch Vụ Đang Tắt", "Dịch vụ Camera ẩn chưa chạy.\n\nHướng dẫn:\n1. Vui lòng mở file 'service_main.exe' trước khi xem live.")
                        msg_root.destroy()
                    threading.Thread(target=show_warn, daemon=True).start()
                    ui.current_state = STATE_MENU
                    continue

                # --- LIVE PREVIEW FROM BACKGROUND SERVICE ---
                from src.config import DATA_DIR
                preview_path = DATA_DIR / "camera_preview.jpg"
                
                # Prepare Display Frame
                display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                
                if os.path.exists(preview_path) and service_active:
                    preview_img = cv2.imread(str(preview_path))
                    if preview_img is not None:
                        p_h, p_w = preview_img.shape[:2]
                        scale = (cur_w - 60) / p_w
                        target_w = int(p_w * scale)
                        target_h = int(p_h * scale)
                        if target_h > cur_h - 200:
                            scale = (cur_h - 220) / p_h
                            target_w = int(p_w * scale)
                            target_h = int(p_h * scale)
                        preview_img = cv2.resize(preview_img, (target_w, target_h))
                        y_off, x_off = 100, (cur_w - target_w) // 2
                        display_frame[y_off:y_off+target_h, x_off:x_off+target_w] = preview_img
                else:
                    cv2.putText(display_frame, "DANG DOI ANH PREVIEW...", (cur_w//2 - 200, cur_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Status Bar inside monitor
                cv2.rectangle(display_frame, (0, 0), (cur_w, 80), (30, 30, 30), -1)
                cv2.putText(display_frame, "GIAM SAT DICH VU CAMERA (SERVICE MONITOR)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.circle(display_frame, (cur_w - 50, 45), 10, (0, 255, 0), -1)
                cv2.putText(display_frame, "[M] Quay ve Menu", (20, cur_h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            elif ui.current_state == STATE_TEST_CAM:
                # 1. Refresh camera config from DB before connecting (User-specific)
                cam_ip = mongo_db.get_setting("camera_ip", CameraConfig.IP, username=ui.session_username)
                cam_port = mongo_db.get_setting("camera_port", CameraConfig.PORT, username=ui.session_username)
                cam_user = mongo_db.get_setting("camera_user", CameraConfig.USER, username=ui.session_username)
                cam_pass = mongo_db.get_setting("camera_pass", CameraConfig.PASS, username=ui.session_username)
                new_url = f"rtsp://{cam_user}:{cam_pass}@{cam_ip}:{cam_port}/ch1/main"
                if cam_ip.isdigit(): new_url = cam_ip # Keep as string for comparison
                
                if str(camera.camera_source) != str(new_url):
                    camera.disconnect()
                    camera = RTSPCamera(rtsp_url=str(new_url))
                
                if not camera.is_connected:
                    if not camera.connect():
                        from tkinter import messagebox
                        import tkinter as tk
                        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
                        messagebox.showerror("Lỗi", f"Không thể kết nối camera tại {cam_ip}!")
                        root.destroy()
                        ui.current_state = STATE_MENU
                        continue
                
                # Inner loop for Direct Test
                while ui.current_state == STATE_TEST_CAM:
                    success, frame = camera.read_frame()
                    if not success or frame is None:
                        f = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        cv2.putText(f, "KHONG THE DOC FRAME", (100, 100), 0, 1, (0,0,255), 2)
                        cv2.imshow(win_name, f)
                        if cv2.waitKey(1) & 0xFF == ord('m'): break
                        continue
                    
                    # Detection every few frames
                    faces = face_rec.detect_and_extract(frame)
                    tracker.update(faces, attendance, frame, company_id=ui.session_company_id)
                    display_frame = face_rec.draw_faces(frame, faces)
                    cv2.putText(display_frame, "CHEDO TEST CAMERA (TRUC TIEP)", (10, cur_h-50), 0, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, "[M] Quay ve Menu", (10, cur_h-20), 0, 0.6, (255,255,255), 1)
                    cv2.imshow(win_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('m'): break
                
                camera.disconnect()
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_ENROLL_CAM:
                if not camera.is_connected:
                    # Show connecting message
                    connecting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(connecting_frame, "Dang ket noi camera...", (400, 360),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                    cv2.imshow(win_name, connecting_frame)
                    cv2.waitKey(1)
                    
                    if not camera.connect():
                        from tkinter import messagebox
                        import tkinter as tk
                        msg_root = tk.Tk()
                        msg_root.withdraw()
                        messagebox.showerror("Lỗi", "Không thể kết nối camera!")
                        msg_root.destroy()
                        ui.current_state = STATE_MENU
                        continue
                
                # Biến cờ để báo hiệu quay lại menu
                enroll_from_camera(camera, face_rec, attendance, ui)
                
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_ENROLL_UPLOAD:
                enroll_by_upload(face_rec, attendance, ui)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_EDIT:
                handle_edit_logic(attendance, face_rec, ui, camera)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LIST:
                # Determine target company with access control
                target_company = get_target_company(ui, mongo_db, allow_selection=True)
                
                if target_company is None:
                    # User cancelled company selection
                    ui.current_state = STATE_MENU
                    continue
                
                # For company users, we also want to see employees from their connected HKB systems
                filter_company = target_company
                if str(ui.session_role).lower() == "company":
                    conns = list(mongo_db.auth_services.find({"user_id": ui.session_user_id}))
                    if conns:
                        filter_company = [target_company] if target_company else []
                        for c in conns:
                            if c["uuid"] not in filter_company:
                                filter_company.append(c["uuid"])
                        logger.info(f"Company user '{ui.session_username}' viewing merged list for IDs: {filter_company}")

                # Get employees from MongoDB (includes all employees, even without face data) for this filter
                mongo_employees = mongo_db.get_all_employees(company_id=filter_company)
                
                # Get employees from Qdrant (only those with face embeddings) for this filter
                qdrant_employees = attendance.get_all_users(company_id=filter_company)
                
                # Merge: prioritize Qdrant data, add MongoDB-only employees
                all_employees = []
                seen_ids = set()
                
                # First add all from Qdrant (have face data)
                for emp in qdrant_employees:
                    u_id_str = str(emp['user_id'])
                    if u_id_str not in seen_ids:
                        all_employees.append({
                            'user_id': u_id_str,
                            'user_name': emp['user_name'],
                            'birthday': emp['birthday'],
                            'has_face': True
                        })
                        seen_ids.add(u_id_str)
                
                # Then add MongoDB-only employees (no face data yet)
                for emp in mongo_employees:
                    # Normalize to string for comparison
                    if str(emp['user_id']) not in seen_ids:
                        all_employees.append({
                            'user_id': emp['user_id'],
                            'user_name': emp['name'],
                            'birthday': emp.get('birthday', 'N/A'),
                            'has_face': False
                        })
                        seen_ids.add(str(emp['user_id']))
                
                logger.info(f"Merged employee list: {len(all_employees)} records found.")
                ui.show_user_list_ui(all_employees)
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_HISTORY:
                # 1. Determine target company with access control
                target_company = get_target_company(ui, mongo_db, allow_selection=True)
                
                if target_company is None:
                    # User cancelled company selection
                    ui.current_state = STATE_MENU
                    continue
                
                # 2. Chọn ngày cần xem
                target_date = ui.get_date_form(title=f"Lịch sử [{target_company}]")
                if not target_date:
                    ui.current_state = STATE_MENU
                    continue

                # 3. Filter history by company and date (company-scoped access)
                logs = mongo_db.get_logs(company_id=target_company, date=target_date)
                
                # Format for display
                display_logs = []
                for l in logs:
                    display_logs.append((
                        str(l["_id"]), l["user_id"], l["user_name"], 
                        l["timestamp"], l["date"], l["status"], 
                        l.get("session_id", str(l["_id"])), 
                        l.get("uploaded_to", [])
                    ))
                
                ui.show_attendance_logs_ui(
                    display_logs, 
                    title=f"Lịch sử ngày {target_date}",
                    session_role=ui.session_role,
                    session_user_id=ui.session_user_id,
                    session_username=ui.session_username
                )
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
                ui.show_system_settings_ui(mongo_db, session_username=ui.session_username)
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
