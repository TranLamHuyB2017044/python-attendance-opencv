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


def get_target_company(ui, mongo_db, allow_selection=True):
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
        logger.info(f"Company user accessing their company: {ui.session_company_id}")
        return ui.session_company_id
    
    # Admin can select company (if allowed) or use their session company
    if str(ui.session_role).lower() == "admin":
        if allow_selection:
            companies = mongo_db.get_all_companies()
            if companies:
                picked = ui.pick_company_ui(companies)
                if picked:
                    logger.info(f"Admin selected company: {picked}")
                    return picked
                else:
                    # User cancelled selection
                    return None
            else:
                # No companies exist, use admin's default
                logger.warning("No companies found, using admin default")
                return ui.session_company_id or MongoDbConfig.COMPANY_ID
        else:
            # Direct use of session company
            return ui.session_company_id or MongoDbConfig.COMPANY_ID
    
    # Fallback to session company or config default
    return ui.session_company_id or MongoDbConfig.COMPANY_ID



def enroll_from_camera(camera, face_rec, attendance, ui):
    """
    Enroll a user by capturing face samples from the camera.
    """
    from src.attendance.mongodb_mgr import MongoDbManager
    from src.config import MongoDbConfig
    mongo_db = MongoDbManager()
    
    user_info = ui.get_user_info_form(
        include_upload=False, 
        session_role=ui.session_role,
        session_company_id=ui.session_company_id,
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
    
    finally:
        # Always cleanup window
        try:
            cv2.destroyWindow("Che do Dang ky")
        except:
            pass


def enroll_by_upload(face_rec, attendance, ui):
    """
    Enroll users by uploading images from disk.
    """
    try:
        # 1. Mở form nhập liệu UI TRƯỚC (có nút chọn ảnh bên trong)
        logger.info("Opening user enrollment form...")
        user_info = AttendanceUI.get_user_form(include_upload=True, session_role=ui.session_role, mongo_db=mongo_db)
        
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
            attendance.upsert_user(u_name, u_id, u_bday, samples, company_id=target_company)
            mongo_db.save_employee(u_id, u_name, u_bday, target_company)
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
                    # Show connecting message
                    connecting_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(connecting_frame, "Dang ket noi camera...", (400, 360),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                    cv2.imshow(win_name, connecting_frame)
                    cv2.waitKey(1)
                    
                    if not camera.connect():
                        # Show error message
                        error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(error_frame, "Khong the ket noi camera!", (350, 340),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                        cv2.putText(error_frame, "Nhan phim bat ky de quay lai menu", (300, 400),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow(win_name, error_frame)
                        cv2.waitKey(2000)  # Show for 2 seconds
                        ui.current_state = STATE_MENU
                        continue
                
                success, frame = camera.read_frame()
                if not success: continue

                from src.config import RecognitionConfig
                faces = face_rec.detect_and_extract(frame, max_faces=RecognitionConfig.MAX_FACES)
                tracker.update(faces, attendance, frame, company_id=ui.session_company_id)
                display_frame = face_rec.draw_faces(frame, faces)
                
                fps_counter += 1
                if time.time() - fps_start_time > 1.0:
                    fps, fps_counter = fps_counter, 0
                    fps_start_time = time.time()
                
                display_frame = ui.draw_status_bar(display_frame, fps, 0) # Time could be added later

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
                # 1. Determine target company with access control
                target_company = get_target_company(ui, mongo_db, allow_selection=True)
                
                if target_company is None:
                    # User cancelled company selection
                    ui.current_state = STATE_MENU
                    continue
                
                # 2. Get merged employee list (MongoDB + Qdrant) for this company ONLY
                mongo_employees = mongo_db.get_all_employees(company_id=target_company)
                qdrant_employees = attendance.get_all_users(company_id=target_company)
                
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
                u_id = ui.pick_user_ui(all_employees)
                
                if u_id:
                    logger.info(f"Selected user_id for edit: {u_id} (type: {type(u_id)})")
                    
                    # 4. Get user info - try Qdrant first, then MongoDB
                    user_info = attendance.get_user_info(u_id)
                    
                    if not user_info:
                        logger.info(f"User {u_id} not found in Qdrant, checking MongoDB...")
                        # Employee exists only in MongoDB, get from there
                        # Convert both to string for comparison
                        mongo_emp = next((e for e in mongo_employees if str(e['user_id']) == str(u_id)), None)
                        if mongo_emp:
                            user_info = {
                                "user_id": mongo_emp['user_id'],
                                "user_name": mongo_emp['name'],
                                "birthday": mongo_emp.get('birthday', 'N/A')
                            }
                            logger.info(f"Found user in MongoDB: {user_info}")
                        else:
                            logger.warning(f"User {u_id} not found in MongoDB either!")
                    else:
                        logger.info(f"Found user in Qdrant: {user_info}")
                    
                    if user_info:
                        edit_res = AttendanceUI.get_edit_user_form(
                            user_info["user_id"], 
                            user_info["user_name"], 
                            user_info["birthday"]
                        )
                        if edit_res:
                            if edit_res["delete"]:
                                # Delete from both Qdrant and MongoDB
                                attendance.delete_user(u_id)
                                mongo_db.employees.delete_one({"user_id": str(u_id), "company_id": target_company})
                                logger.success(f"Da xoa nhan vien ID: {u_id}")
                            
                            elif edit_res["enroll_camera"]:
                                # Update info first, then enroll via camera
                                attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                                mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company)
                                logger.info(f"Updated info for {u_id}, starting camera enrollment...")
                                
                                # Trigger camera enrollment
                                if not camera.is_connected:
                                    camera.connect()
                                
                                # Call enrollment function
                                samples = []
                                logger.info(f"Collecting 3-5 samples for '{edit_res['name']}' (ID: {u_id}). Press 's' to capture, 'f' to finish, 'm' to menu, 'c' to cancel.")
                                
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
                                            break
                                        elif key == ord('c'):
                                            logger.warning("Enrollment cancelled by user.")
                                            break

                                    if len(samples) >= 1:
                                        attendance.upsert_user(edit_res["name"], u_id, edit_res["bday"], samples, company_id=target_company)
                                        logger.success(f"Da dang ky khuon mat cho: {edit_res['name']} (ID: {u_id})")
                                        
                                        from tkinter import messagebox
                                        import tkinter as tk
                                        msg_root = tk.Tk()
                                        msg_root.withdraw()
                                        messagebox.showinfo("Thành công", f"Đã đăng ký khuôn mặt cho {edit_res['name']}")
                                        msg_root.destroy()
                                
                                finally:
                                    # Always cleanup window
                                    try:
                                        cv2.destroyWindow("Che do Dang ky")
                                    except:
                                        pass
                            
                            elif edit_res["enroll_upload"]:
                                # Update info first, then enroll via file upload
                                attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                                mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company)
                                logger.info(f"Updated info for {u_id}, starting file upload enrollment...")
                                
                                # Trigger file upload enrollment
                                from tkinter import filedialog
                                import tkinter as tk
                                
                                file_root = tk.Tk()
                                file_root.withdraw()
                                file_paths = filedialog.askopenfilenames(
                                    title="Chọn ảnh khuôn mặt (Ít nhất 1 ảnh)",
                                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
                                )
                                file_root.destroy()
                                
                                if file_paths and len(file_paths) >= 1:
                                    samples = []
                                    for fpath in file_paths[:5]:
                                        img = cv2.imread(fpath)
                                        if img is not None:
                                            faces = face_rec.detect_and_extract(img)
                                            if faces:
                                                samples.append(faces[0].normed_embedding)
                                                logger.info(f"Extracted face from {fpath}")
                                    
                                    if len(samples) >= 1:
                                        attendance.upsert_user(edit_res["name"], u_id, edit_res["bday"], samples, company_id=target_company)
                                        logger.success(f"Da dang ky khuon mat cho: {edit_res['name']} (ID: {u_id})")
                                        
                                        from tkinter import messagebox
                                        import tkinter as tk
                                        msg_root = tk.Tk()
                                        msg_root.withdraw()
                                        messagebox.showinfo("Thành công", f"Đã đăng ký {len(samples)} ảnh khuôn mặt cho {edit_res['name']}")
                                        msg_root.destroy()
                                    else:
                                        from tkinter import messagebox
                                        import tkinter as tk
                                        msg_root = tk.Tk()
                                        msg_root.withdraw()
                                        messagebox.showwarning("Cảnh báo", "Không tìm thấy khuôn mặt hợp lệ trong các ảnh đã chọn!")
                                        msg_root.destroy()
                            
                            else:
                                # Just update info, no enrollment
                                attendance.update_user_info(u_id, edit_res["name"], edit_res["bday"])
                                mongo_db.save_employee(str(u_id), edit_res["name"], edit_res["bday"], target_company)
                                logger.success(f"Da cap nhat thong tin nhan vien ID: {u_id}")
                    else:
                        logger.error(f"Could not find user info for {u_id}")
                        from tkinter import messagebox
                        messagebox.showerror("Lỗi", f"Không tìm thấy thông tin nhân viên {u_id}")
                
                ui.current_state = STATE_MENU
                continue

            elif ui.current_state == STATE_LIST:
                # Determine target company with access control
                target_company = get_target_company(ui, mongo_db, allow_selection=True)
                
                if target_company is None:
                    # User cancelled company selection
                    ui.current_state = STATE_MENU
                    continue
                
                # Get employees from MongoDB (includes all employees, even without face data) for this company ONLY
                mongo_employees = mongo_db.get_all_employees(company_id=target_company)
                
                # Get employees from Qdrant (only those with face embeddings) for this company ONLY
                qdrant_employees = attendance.get_all_users(company_id=target_company)
                
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
                    session_user_id=ui.session_user_id
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
