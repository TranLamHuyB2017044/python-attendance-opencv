import cv2
import time
import os
import sys
import warnings
import numpy as np
from loguru import logger
import tkinter as tk
from tkinter import filedialog

# --- FIX: ADD PROJECT ROOT TO PATH ---
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    # Get the parent directory of 'src'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)
# -------------------------------------

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
from src.services.ping_service import ping_service
from src.utils.log_panel_renderer import draw_log_panel
from src.utils.webhook_log_bus import start_polling


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
    
    user_id, user_name, birthday, _, selected_cid, user_gender = user_info

    samples = []
    enrollment_image_ids = []
    logger.info(f"Collecting 3-5 samples for '{user_name}' (ID: {user_id}). Press 's' to capture, 'f' to finish, 'm' to menu, 'c' to cancel.")
    
    target_company = MongoDbConfig.COMPANY_ID
    # Use selected company if admin, otherwise session company
    if str(ui.session_role).lower() == 'admin' and selected_cid:
        target_company = selected_cid
    elif ui.session_company_id:
        target_company = ui.session_company_id

    try:
        while len(samples) < 10:
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
                cv2.putText(display_frame, f"Mau {len(samples)}/10", (10, 30),
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
                    # Save the frame to MongoDB enrollment_images
                    success, encoded_img = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                    if success:
                        image_blob = encoded_img.tobytes()
                        image_id = mongo_db.save_enrollment_image(user_id, target_company, image_blob)
                        if image_id:
                            enrollment_image_ids.append(image_id)
                    logger.info(f"Captured sample {len(samples)}/10")
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

        if len(samples) >= 1:
            # --- Check for existing user to ask before update ---
            force_upd = False
            existing = mongo_db.employees.find_one({"user_id": str(user_id), "company_id": target_company})
            if existing:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
                if messagebox.askyesno("Xác nhận", f"Mã nhân viên '{user_id}' đã tồn tại trong hệ thống.\n\nBạn có muốn CẬP NHẬT dữ liệu mới nhất cho nhân viên này không?"):
                    force_upd = True
                else:
                    logger.info("User cancelled update.")
                    root.destroy()
                    return
                root.destroy()

            # Try to save to MongoDB
            ok, msg = mongo_db.save_employee(user_id, user_name, birthday, target_company, sex=user_gender, force_update=force_upd)
            if not ok:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
                messagebox.showerror("Lỗi đăng ký", f"Không thể lưu nhân viên: {msg}")
                root.destroy()
                return

            ok_qdrant = attendance.upsert_user(user_name, user_id, birthday, samples, clear_old=force_upd, company_id=target_company, enrollment_image_ids=enrollment_image_ids)
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
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
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
            
        u_id, u_name, u_bday, file_paths, selected_cid, u_gender = user_info
        logger.info(f"Processing enrollment for {u_name} (ID: {u_id}) with {len(file_paths)} files.")
        
        target_company = MongoDbConfig.COMPANY_ID
        if str(ui.session_role).lower() == 'admin' and selected_cid:
            target_company = selected_cid
        elif ui.session_company_id:
            target_company = ui.session_company_id

        samples = []
        enrollment_image_ids = []
        for i, path in enumerate(file_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(file_paths)}: {path}")
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.error(f"Could not read image: {path}")
                    continue
                    
                faces = face_rec.detect_and_extract(img)
                if faces:
                    # Sort by face size to get the most prominent face
                    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    samples.append(faces[0].normed_embedding)
                    
                    # Save to MongoDB
                    success, encoded_img = cv2.imencode('.webp', img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                    if success:
                        image_blob = encoded_img.tobytes()
                        image_id = mongo_db.save_enrollment_image(u_id, target_company, image_blob, image_path=path)
                        if image_id:
                            enrollment_image_ids.append(image_id)
                            
                    logger.info(f"Successfully extracted face embedding from: {path}")
                else:
                    logger.warning(f"No face detected in image: {path}")
            except Exception as img_err:
                logger.error(f"Error processing image {path}: {img_err}")

        if samples:
            logger.info(f"Face extraction complete. {len(samples)} valid samples found.")
            
            # Update databases
            logger.info(f"Saving to Qdrant and MongoDB for company: {target_company}")
            
            # --- Check for existing user to ask before update ---
            force_upd = False
            existing = mongo_db.employees.find_one({"user_id": str(u_id), "company_id": target_company})
            if existing:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
                if messagebox.askyesno("Xác nhận", f"Mã nhân viên '{u_id}' đã tồn tại trong hệ thống.\n\nBạn có muốn CẬP NHẬT dữ liệu mới nhất cho nhân viên này không?"):
                    force_upd = True
                else:
                    logger.info("User cancelled update.")
                    root.destroy()
                    return
                root.destroy()

            # Validation via MongoDB
            ok, msg = mongo_db.save_employee(u_id, u_name, u_bday, target_company, sex=u_gender, force_update=force_upd)
            if not ok:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
                messagebox.showerror("Lỗi đăng ký", f"Không thể lưu nhân viên: {msg}")
                root.destroy()
                return

            attendance.upsert_user(u_name, u_id, u_bday, samples, clear_old=force_upd, company_id=target_company, enrollment_image_ids=enrollment_image_ids)
            logger.success(f"Successfully enrolled {u_name} via upload.")
            
            # Show success message using a robust method
            try:
                from tkinter import messagebox
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", False)
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

    # 2. Start editing loop
    while True:
        # Get merged employee list (MongoDB + Qdrant) for this company ONLY (Only Active Users)
        mongo_employees = mongo_db.get_all_employees(company_id=filter_company, active_only=True)
        qdrant_employees = attendance.get_all_users(company_id=filter_company, active_only=True)

        
        # Merge employee lists
        all_employees = []
        seen_ids = set()
        
        # Create a map for mongo employees for easy lookup
        mongo_map = {str(emp['user_id']): emp for emp in mongo_employees}
        
        # First add all from Qdrant (have face data)
        for emp in qdrant_employees:
            u_id_str = str(emp['user_id'])
            if u_id_str not in seen_ids:
                # Get sex from mongo if possible
                mongo_emp = mongo_map.get(u_id_str)
                sex = mongo_emp.get('sex', 'Nam') if mongo_emp else 'Nam'
                
                all_employees.append({
                    'user_id': u_id_str,
                    'user_name': emp['user_name'],
                    'birthday': emp['birthday'],
                    'sex': sex,
                    'has_face': True,
                    'active': emp.get('active', True)
                })
                seen_ids.add(u_id_str)
        
        # Then add MongoDB-only employees (no face data yet)
        for emp in mongo_employees:
            u_id_str = str(emp['user_id'])
            # Normalize to string for comparison
            if u_id_str not in seen_ids:
                all_employees.append({
                    'user_id': emp['user_id'],
                    'user_name': emp['name'],
                    'birthday': emp.get('birthday', 'N/A'),
                    'sex': emp.get('sex', 'Nam'),
                    'has_face': False,
                    'active': emp.get('active', True)
                })
                seen_ids.add(u_id_str)
        
        # 3. Pick User from merged list
        u_id = ui.pick_user_ui(all_employees, parent=parent)
        
        if not u_id:
            logger.info("No user selected or selection cancelled. Exiting edit mode.")
            break
            
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
                    "birthday": mongo_emp.get('birthday', 'N/A'),
                    "sex": mongo_emp.get('sex', 'Nam')
                }
            else:
                logger.warning(f"User {u_id} not found in MongoDB either!")
        else:
            # If found in Qdrant, still check MongoDB for sex/gender
            mongo_emp = next((e for e in mongo_employees if str(e['user_id']) == str(u_id)), None)
            if mongo_emp:
                user_info["sex"] = mongo_emp.get("sex", "Nam")
            else:
                user_info["sex"] = "Nam"
        
        if user_info:
            edit_res = ui.get_edit_user_form(
                user_info["user_id"], 
                user_info["user_name"], 
                user_info["birthday"],
                current_sex=user_info.get("sex", "Nam"),
                session_role=ui.session_role,
                parent=parent
            )
            if edit_res:
                if edit_res["delete"]:
                    # Soft delete from both Qdrant and MongoDB
                    attendance.set_user_active_status(u_id, False)
                    mongo_db.soft_delete_employee(str(u_id), target_company)
                    logger.success(f"Da xoa mem nhan vien ID: {u_id}")
                
                elif edit_res["enroll_camera"]:
                    # Update thong tin trước, sau đó dang ký qua camera
                    attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company, sex=edit_res["sex"], force_update=True)
                    
                    if not camera.is_connected: camera.connect()
                    
                    # Trigger collection
                    samples = []
                    enrollment_image_ids = []
                    try:
                        while len(samples) < 10:
                            success, frame = camera.read_frame()
                            if not success or frame is None: continue
                            
                            display_frame = frame.copy()
                            faces = face_rec.detect_and_extract(frame)
                            h, w = display_frame.shape[:2]
                            
                            if faces:
                                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                                bbox = faces[0].bbox.astype(int)
                                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                                cv2.putText(display_frame, f"Mau {len(samples)}/10", (10, 30), 0, 0.8, (255, 255, 0), 2)
                            else:
                                cv2.putText(display_frame, "Khong tim thay mat!", (10, 30), 0, 0.8, (0, 0, 255), 2)
                            
                            cv2.rectangle(display_frame, (0, h-60), (w, h), (0, 0, 0), -1)
                            cv2.putText(display_frame, "[S] Luu mau  [F] Hoan thanh  [M] Menu  [C] Huy", (10, h-20), 0, 0.7, (255, 255, 255), 2)
                            cv2.imshow("Che do Dang ky", display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('s'):
                                if faces:
                                    samples.append(faces[0].normed_embedding)
                                    # Save frame to MongoDB enrollment_images
                                    enc_ok, enc_img = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                                    if enc_ok:
                                        img_id = mongo_db.save_enrollment_image(str(u_id), target_company, enc_img.tobytes())
                                        if img_id:
                                            enrollment_image_ids.append(img_id)
                                    logger.info(f"Captured sample {len(samples)}/10")
                            elif key == ord('f') and len(samples) >= 1: break
                            elif key in [ord('m'), ord('c')]: break
                        
                        if len(samples) >= 1:
                            attendance.upsert_user(edit_res["name"], u_id, edit_res["bday"], samples, clear_old=True, company_id=target_company, enrollment_image_ids=enrollment_image_ids)
                            mongo_db.update_employee_has_face(str(u_id), True)
                            from tkinter import messagebox
                            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
                            messagebox.showinfo("Thành công", f"Đã đăng ký khuôn mặt cho {edit_res['name']}")
                            root.destroy()
                    finally:
                        try: cv2.destroyWindow("Che do Dang ky")
                        except: pass
                
                elif edit_res["enroll_upload"]:
                    # Update thong tin truoc
                    attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company, sex=edit_res["sex"], force_update=True)
                    
                    from tkinter import filedialog
                    # Dùng parent window (CTk dashboard) thay vì tạo tk.Tk() mới,
                    # vì tạo tk.Tk() mới khi event loop CTk đang chạy sẽ làm filedialog
                    # mở ra rồi tự đóng ngay lập tức.
                    _dialog_parent = parent  # có thể là CTkToplevel hoặc None
                    if _dialog_parent is None:
                        _tmp_root = tk.Tk()
                        _tmp_root.withdraw()
                        _tmp_root.attributes("-topmost", False)
                        _dialog_parent = _tmp_root
                    else:
                        _tmp_root = None
                    
                    file_paths = filedialog.askopenfilenames(
                        title="Chọn ảnh khuôn mặt",
                        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")],
                        parent=_dialog_parent
                    )
                    
                    if _tmp_root is not None:
                        _tmp_root.destroy()
                    
                    if file_paths:
                        samples = []
                        enrollment_image_ids = []
                        for fp in file_paths[:10]:
                            img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                faces = face_rec.detect_and_extract(img)
                                if faces:
                                    samples.append(faces[0].normed_embedding)
                                    # Save image to MongoDB enrollment_images
                                    enc_ok, enc_img = cv2.imencode('.webp', img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                                    if enc_ok:
                                        img_id = mongo_db.save_enrollment_image(str(u_id), target_company, enc_img.tobytes(), image_path=fp)
                                        if img_id:
                                            enrollment_image_ids.append(img_id)
                        
                        if len(samples) >= 1:
                            attendance.upsert_user(edit_res["name"], u_id, edit_res["bday"], samples, clear_old=True, company_id=target_company, enrollment_image_ids=enrollment_image_ids)
                            mongo_db.update_employee_has_face(str(u_id), True)
                            from tkinter import messagebox
                            if parent:
                                messagebox.showinfo("Thành công", f"Đã cập nhật {len(samples)} ảnh cho {edit_res['name']}", parent=parent)
                            else:
                                _msg_root = tk.Tk(); _msg_root.withdraw(); _msg_root.attributes("-topmost", False)
                                messagebox.showinfo("Thành công", f"Đã cập nhật {len(samples)} ảnh cho {edit_res['name']}")
                                _msg_root.destroy()
                else:
                    # Only update info
                    mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company, sex=edit_res["sex"], force_update=True)
                    attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                    logger.success(f"Da cap nhat thong tin ID: {u_id}")
        else:
            from tkinter import messagebox
            messagebox.showerror("Lỗi", "Không tìm thấy thông tin nhân viên")


def cleanup_old_videos(days=60):
    """Xóa video cũ hơn 60 ngày để giải phóng ổ cứng."""
    try:
        from src.config import CAPTURES_DIR
        import time
        import os
        
        now = time.time()
        max_age = days * 24 * 3600
        count = 0
        
        if not CAPTURES_DIR.exists():
            return
            
        for root, dirs, files in os.walk(CAPTURES_DIR):
            for file in files:
                if file.endswith((".mp4", ".webm", ".avi")):
                    file_path = os.path.join(root, file)
                    try:
                        file_age = now - os.path.getmtime(file_path)
                        if file_age > max_age:
                            os.remove(file_path)
                            count += 1
                    except Exception:
                        continue
        if count > 0:
            logger.info(f"Cleanup: Da xoa {count} video cu hon {days} ngay.")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def main():
    setup_logger()
    logger.info("Initializing Face Attendance System...")

    # Ping Service is disabled for Management GUI to avoid double reporting on Dashboard
    # ping_service.start()

    # Load config from MongoDB so that Cooldown/Anti-Spoofing settings take effect immediately
    CameraConfig.load_from_mongodb(mongo_db)

    # Auto cleanup old videos in background
    import threading
    threading.Thread(target=cleanup_old_videos, args=(60,), daemon=True).start()

    # Start webhook log polling
    start_polling()

    try:
        face_rec = FaceRecognition()
        attendance = QdrantAttendanceManager()
        # GUI app doesn't send telegram notifications to avoid spamming
        camera = RTSPCamera(enable_notifications=False)
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
    main.last_ai_time_detect = 0
    main.cached_faces_detect = []

    display_frame = ui.draw_main_menu()
    cv2.imshow(win_name, display_frame)
    if not ui.show_login_dialog():
        logger.warning("Truy cap bi tu choi hoặc ứng dụng bị đóng.")
        return

    try:
        service_active = False
        last_heartbeat_check = 0
        last_w, last_h = 0, 0
        
        # --- DECOUPLED AI BACKGROUND THREAD FOR GUI ---
        import queue
        import threading
        main.ai_queue = queue.Queue(maxsize=1)
        main.cached_faces_detect = []
        
        def gui_ai_worker():
            import time
            import numpy as np
            while True:
                try:
                    frame_data = main.ai_queue.get()
                    if frame_data is None: 
                        break # Stop signal
                    frame, company_id = frame_data
                    
                    ai_frame_obj = frame.copy()
                    if CameraConfig.ROI:
                        x1, y1, x2, y2 = CameraConfig.ROI
                        mask = np.zeros_like(ai_frame_obj)
                        h, w = ai_frame_obj.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        mask[y1:y2, x1:x2] = 255
                        ai_frame_obj = cv2.bitwise_and(ai_frame_obj, mask)
                        
                    faces = face_rec.detect_and_extract(ai_frame_obj, fast=True)
                    tracker.update(faces, attendance, face_rec=face_rec, frame=frame, company_id=company_id)
                    main.cached_faces_detect = faces
                    
                    time.sleep(0.01) # Small sleep to yield CPU
                except Exception as e:
                    logger.debug(f"GUI AI Worker error: {e}")
                finally:
                    if frame_data is not None:
                        main.ai_queue.task_done()
                        
        ai_thread = threading.Thread(target=gui_ai_worker, daemon=True)
        ai_thread.start()
        
        while True:
            # Get actual window size for responsive drawing (with safety checks)
            try:
                # Check if window exists first
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    # If window was closed by user, default to menu or reset
                    if ui.current_state not in [STATE_MENU]:
                        ui.current_state = STATE_MENU
                        continue
                    cur_w, cur_h = 1280, 720
                else:
                    _, _, cur_w, cur_h = cv2.getWindowImageRect(win_name)
                    if cur_w <= 0 or cur_h <= 0:
                        cur_w, cur_h = 1280, 720
            except:
                # Fallback if window is being destroyed or not yet created
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
                # Close OpenCV window temporarily to show modern Dashboard
                cv2.destroyWindow(win_name)
                # This call is BLOCKING until a button is clicked or window closed
                next_state = ui.show_main_dashboard(mongo_db, attendance=attendance, face_rec=face_rec, camera=camera, service_active=service_active)
                ui.current_state = next_state
                # Re-create window for other states
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 1280, 720)
                cv2.setMouseCallback(win_name, ui.handle_menu_click, param=(cur_w, cur_h))
                main._detect_callback_cleared = False  # Reset để lần sau vào DETECT lại xóa callback
                last_w, last_h = 1280, 720
                continue
                
            elif ui.current_state == STATE_DETECT:
                # Xóa mouse callback MỘT LẦN khi mới vào STATE_DETECT
                # → Ngăn click vào camera view vô tình trigger menu buttons
                if not getattr(main, '_detect_callback_cleared', False):
                    def on_panel_click(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            # Tọa độ x0, pw, ph được truyền vào param hoặc tính toán dựa trên current state
                            # CLEAR button trong log_panel_renderer: [x0+pw-50, x0+pw-4], [ph-17, ph-3]
                            if RecognitionConfig.TEST_MODE:
                                cur_w, cur_h = param
                                p_w = max(220, int(cur_w * 0.25))
                                x0 = cur_w - p_w
                                if x >= x0 + p_w - 55 and y >= cur_h - 22:
                                    from src.utils.webhook_log_bus import clear_logs
                                    clear_logs()

                    try:
                        cv2.setMouseCallback(win_name, on_panel_click, param=(cur_w, cur_h))
                    except Exception:
                        pass
                    main._detect_callback_cleared = True

                # Layout:  75% camera | 25% log panel (Only in TEST_MODE)
                from src.config import RecognitionConfig
                if RecognitionConfig.TEST_MODE:
                    panel_w    = max(220, int(cur_w * 0.25))
                    cam_area_w = cur_w - panel_w
                else:
                    panel_w    = 0
                    cam_area_w = cur_w

                # Update click area dynamically (important if window resized)
                if not getattr(main, '_detect_callback_cleared', False):
                    # callback set up elsewhere, but we ensure it uses these values
                    pass

                # --- AUTO SWITCH MODE: DIRECT (DEV) vs PREVIEW (PROD/EXE) ---
                if getattr(sys, 'frozen', False):
                    # --- PRODUCTION MODE: SHOW PREVIEW FROM SERVICE (SHARED MEMORY) ---
                    from multiprocessing import shared_memory
                    SHM_NAME = "bittech_monitor_shm"
                    SHM_SIZE_MAX = 5 * 1024 * 1024
                    display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                    
                    shm_frame = None
                    try:
                        existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
                        seq1 = int(existing_shm.buf[0])
                        # EVEN = frame hợp lệ (ghi xong), bỏ seq1 > 0 vì seq=0 là EVEN hợp lệ
                        if seq1 % 2 == 0:
                            seq2 = int(existing_shm.buf[0])
                            if seq1 == seq2 and existing_shm.buf[1] == 1:
                                w = int(np.frombuffer(existing_shm.buf[2:4], dtype=np.uint16)[0])
                                h = int(np.frombuffer(existing_shm.buf[4:6], dtype=np.uint16)[0])
                                if 0 < w < 4000 and 0 < h < 4000:
                                    size = w * h * 3
                                    if 10 + size <= SHM_SIZE_MAX:
                                        data = bytes(existing_shm.buf[10:10+size])
                                        shm_frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                        existing_shm.close()
                    except Exception:
                        pass

                    if shm_frame is not None:
                        p_h, p_w = shm_frame.shape[:2]
                        # Use cam_area_w instead of full cur_w
                        scale = (cam_area_w - 60) / p_w
                        target_w, target_h = int(p_w * scale), int(p_h * scale)
                        if target_h > cur_h - 180:
                            scale = (cur_h - 200) / p_h
                            target_w, target_h = int(p_w * scale), int(p_h * scale)
                        
                        try:
                            resized_preview = cv2.resize(shm_frame, (target_w, target_h))
                            y_off, x_off = 80, (cam_area_w - target_w) // 2
                            display_frame[y_off:y_off+target_h, x_off:x_off+target_w] = resized_preview
                        except: pass
                        
                        # Drawn atop Cam area
                        cv2.rectangle(display_frame, (0, 0), (cam_area_w, 40), (40, 40, 40), -1)
                        cv2.putText(display_frame, "SERVICE MONITOR (PRODUCTION MODE)", (20, 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "DANG DOI KET NOI VOI SERVICE...", (cam_area_w//2 - 250, cur_h//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # --- DEVELOPMENT MODE: DIRECT CAMERA CONNECTION FOR TESTING ---
                    if getattr(ui, 'use_local_webcam', False):
                        new_url = "0"
                    else:
                        cam_ip = mongo_db.get_setting("camera_ip", CameraConfig.IP, username=ui.session_username)
                        cam_port = mongo_db.get_setting("camera_port", CameraConfig.PORT, username=ui.session_username)
                        cam_user = mongo_db.get_setting("camera_user", CameraConfig.USER, username=ui.session_username)
                        cam_pass = mongo_db.get_setting("camera_pass", CameraConfig.PASS, username=ui.session_username)
                        
                        # Use URL directly from centralized CameraConfig
                        # CameraConfig already handles priority between .env and MongoDB
                        new_url = CameraConfig.RTSP_URL
                        
                        if str(cam_ip).isdigit(): new_url = str(cam_ip)
                    
                    if str(camera.camera_source) != str(new_url):
                        camera.disconnect()
                        # Reuse the notification-disabled setting
                        camera = RTSPCamera(rtsp_url=str(new_url), enable_notifications=False)

                    if not camera.is_connected:
                        if not camera.connect():
                            ui.current_state = STATE_MENU
                            continue

                    success, frame = camera.read_frame()
                    if not success or frame is None:
                        display_frame = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "DANG DOC DU LIEU TU CAMERA...", (cur_w//2 - 200, cur_h//2), 0, 0.7, (0, 255, 255), 2)
                    else:
                        # Update FPS
                        fps_counter += 1
                        if time.time() - fps_start_time > 1.0:
                            fps = fps_counter / (time.time() - fps_start_time)
                            fps_counter = 0
                            fps_start_time = time.time()

                        # Push to AI thread asynchronously (Non-blocking)
                        if not main.ai_queue.full():
                            try:
                                main.ai_queue.put_nowait((frame.copy(), ui.session_company_id))
                            except Exception:
                                pass

                        # Always use the latest available AI results
                        faces = main.cached_faces_detect

                        display_frame = face_rec.draw_faces(frame, faces)

                        # ── Always Resize & Center onto Canvas ──
                        f_h, f_w = display_frame.shape[:2]
                        scale = min(cam_area_w / f_w, cur_h / f_h)
                        cur_disp_h, cur_disp_w = int(f_h * scale), int(f_w * scale)
                        
                        cam_resized = cv2.resize(display_frame, (cur_disp_w, cur_disp_h))
                        
                        canvas = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        y_off = (cur_h - cur_disp_h) // 2
                        x_off = (cam_area_w - cur_disp_w) // 2
                        canvas[y_off:y_off+cur_disp_h, x_off:x_off+cur_disp_w] = cam_resized
                        display_frame = canvas

                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (cam_area_w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # --- START UI OVERLAY ---
                        current_t = time.time()
                        # 1. Ảnh Check-in thành công (Bên Trái) - Tồn tại 3 giây và mờ dần
                        tracker.recent_snapshots = [s for s in getattr(tracker, 'recent_snapshots', []) if current_t - s["time"] < 3.0]
                        
                        overlay_x = 20
                        overlay_y = 100
                        for snap in tracker.recent_snapshots:
                            thumb_w, thumb_h = 160, 90
                            thumb = cv2.resize(snap["img"], (thumb_w, thumb_h))
                            
                            box_x = overlay_x
                            box_y = overlay_y - 20
                            box_w = thumb_w
                            box_h = thumb_h + 20
                            
                            if box_x + box_w < cam_area_w and box_y + box_h < cur_h and box_x >= 0 and box_y >= 0:
                                try:
                                    elapsed = current_t - snap["time"]
                                    if elapsed < 1.0:
                                        alpha = 1.0 # 1 giây đầu hiển thị rõ 100%
                                    else:
                                        alpha = max(0.0, 1.0 - (elapsed - 1.0) / 2.0) # 2 giây sau mờ dần đi (Fade out)
                                    
                                    bg = display_frame[box_y:box_y+box_h, box_x:box_x+box_w].copy()
                                    fg = bg.copy()
                                    
                                    # Vẽ đè lên background của region này
                                    color = (0, 255, 0)
                                    fg[20:20+thumb_h, 0:thumb_w] = thumb
                                    cv2.rectangle(fg, (0, 20), (thumb_w, 20+thumb_h), color, 2)
                                    cv2.putText(fg, snap.get("name", "Unknown"), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                                    # Trộn ảnh theo Alpha
                                    blended = cv2.addWeighted(fg, alpha, bg, 1.0 - alpha, 0)
                                    display_frame[box_y:box_y+box_h, box_x:box_x+box_w] = blended
                                    
                                except Exception:
                                    pass
                                overlay_y += box_h + 30
                        # --- END UI OVERLAY ---
                        
                        if CameraConfig.ROI:
                            x1, y1, x2, y2 = CameraConfig.ROI
                            # Since display_frame is now the full canvas, mapping ROI requires offset
                            rx1 = x_off + int(x1 * scale)
                            ry1 = y_off + int(y1 * scale)
                            rx2 = x_off + int(x2 * scale)
                            ry2 = y_off + int(y2 * scale)
                            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 120, 0), 2)
                            cv2.putText(display_frame, "VUNG CHAM CONG", (rx1 + 10, ry1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)

                # ── Log panel ────────────────────────────────────────────────────────
                if panel_w > 0:
                    draw_log_panel(display_frame, cam_area_w, panel_w, cur_h)

                cv2.putText(display_frame, "[M] Thoat ve Menu", (20, cur_h - 20), 0, 0.6, (200, 200, 200), 1)

            elif ui.current_state == STATE_TEST_CAM:
                # 1. Refresh camera config from DB before connecting (User-specific)
                cam_ip = mongo_db.get_setting("camera_ip", CameraConfig.IP, username=ui.session_username)
                cam_port = mongo_db.get_setting("camera_port", CameraConfig.PORT, username=ui.session_username)
                cam_user = mongo_db.get_setting("camera_user", CameraConfig.USER, username=ui.session_username)
                cam_pass = mongo_db.get_setting("camera_pass", CameraConfig.PASS, username=ui.session_username)
                
                # Use centralized config as single source of truth
                new_url = CameraConfig.RTSP_URL
                
                if str(cam_ip).isdigit(): new_url = cam_ip # Keep as string for comparison
                
                if str(camera.camera_source) != str(new_url):
                    camera.disconnect()
                    camera = RTSPCamera(rtsp_url=str(new_url), enable_notifications=False)
                
                if not camera.is_connected:
                    if not camera.connect():
                        from tkinter import messagebox
                        import tkinter as tk
                        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", False)
                        messagebox.showerror("Lỗi", f"Không thể kết nối camera tại {cam_ip}!")
                        root.destroy()
                        ui.current_state = STATE_MENU
                        continue
                
                # Inner loop for Direct Test
                fps_test_start = time.time()
                fps_test_counter = 0
                fps_test = 0
                while ui.current_state == STATE_TEST_CAM:
                    success, frame = camera.read_frame()
                    if not success or frame is None:
                        f = np.zeros((cur_h, cur_w, 3), dtype=np.uint8)
                        cv2.putText(f, "KHONG THE DOC FRAME", (100, 100), 0, 1, (0,0,255), 2)
                        cv2.imshow(win_name, f)
                        if cv2.waitKey(1) & 0xFF == ord('m'): break
                        continue
                    
                    # Update FPS display variables
                    fps_test_counter += 1
                    if time.time() - fps_test_start > 1.0:
                        fps_test = fps_test_counter / (time.time() - fps_test_start)
                        fps_test_counter = 0
                        fps_test_start = time.time()

                    # Push to AI thread asynchronously (Non-blocking)
                    if not main.ai_queue.full():
                        try:
                            main.ai_queue.put_nowait((frame.copy(), ui.session_company_id))
                        except Exception:
                            pass
                    
                    # Always use the latest available AI results
                    faces = main.cached_faces_detect
                    
                    display_frame = face_rec.draw_faces(frame, faces)
                    cv2.putText(display_frame, f"FPS: {fps_test:.1f}", (cur_w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if CameraConfig.ROI:
                        x1, y1, x2, y2 = CameraConfig.ROI
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 120, 0), 3)
                        cv2.putText(display_frame, "VUNG CHAM CONG", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)
                        
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
                mongo_employees = mongo_db.get_all_employees(company_id=filter_company, active_only=False)
                
                # Get employees from Qdrant (only those with face embeddings) for this filter
                qdrant_employees = attendance.get_all_users(company_id=filter_company, active_only=False)
                
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
                            'has_face': True,
                            'active': emp.get('active', True)
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
                            'has_face': False,
                            'active': emp.get('active', True)
                        })
                        seen_ids.add(str(emp['user_id']))
                
                logger.info(f"Merged employee list: {len(all_employees)} records found.")
                ui.show_user_list_ui(all_employees, company_id=target_company, mongo_db=mongo_db)
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
                company_displayName = mongo_db.get_company_name(target_company)
                target_date = ui.get_date_form(title=f"Lịch sử [{company_displayName}]", ok_button_text="LẤY DỮ LIỆU")
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
                    logs, 
                    title=f"Lịch sử ngày {target_date}",
                    session_role=ui.session_role,
                    session_user_id=ui.session_user_id,
                    session_username=ui.session_username,
                    mongo_db=mongo_db,
                    target_company=target_company
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
        ping_service.stop()  # Dừng Ping Service khi app tắt
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
