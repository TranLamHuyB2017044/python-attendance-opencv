import cv2
import numpy as np
import os
import sys

# --- FIX: ADD PROJECT ROOT TO PATH ---
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    # Get the parent directory of 'src'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)
# -------------------------------------

from loguru import logger
from src.attendance.mongodb_mgr import mongo_db
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
from src.ui.date_picker import CTkDatePicker, CTkDateEntry

def _apply_icon(window):
    """Utility to apply app icon to any tkinter/customtkinter window."""
    try:
        # Check if running as a bundled executable
        if getattr(sys, 'frozen', False):
            icon_path = os.path.join(sys._MEIPASS, "app_icon.ico")
        else:
            icon_path = "app_icon.ico"
            
        if os.path.exists(icon_path):
            window.iconbitmap(icon_path)
    except Exception:
        pass

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# State Management Constants
STATE_MENU = 0
STATE_DETECT = 1
STATE_ENROLL_CAM = 2
STATE_ENROLL_UPLOAD = 3
STATE_EDIT = 4
STATE_LIST = 5
STATE_HISTORY = 6
STATE_HKB_LIST = 7
STATE_COMPANY = 8
STATE_CLOUD_USER = 9
STATE_LOGOUT = 10
STATE_SETTINGS = 11
STATE_TEST_CAM = 12
STATE_EXIT = 99

class AttendanceUI:
    """
    Handles all UI rendering and interaction logic.
    """
    def __init__(self):
        self.current_state = STATE_MENU
        self.is_admin_logged_in = False
        self.session_role = None # 'admin' or 'company'
        self.session_company_id = None
        self.session_username = None
        self.session_user_id = 1 # ID mặc định cho system user
        self.service_active = False # Track if background service is running
        self.last_w, self.last_h = 1280, 720 # Default

    def handle_menu_click(self, event, x, y, flags, param):
        """Handle mouse clicks for the menu."""
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = param
            cX, cY = w // 2, h // 2
            
            # Use ratios if window was resized
            rW, rH = w / 800, h / 600 # Legacy base logic or dynamic
            
            # To simplify, we'll use the same dynamic logic as drawing
            btn_w, btn_h = int(300 * (w/800)), int(60 * (h/600))
            gap_x = int(10 * (w/800))
            gap_y = int(20 * (h/600))
            col1_L, col1_R = cX - btn_w - gap_x, cX - gap_x
            col2_L, col2_R = cX + gap_x, cX + gap_x + btn_w

            # 0. DANG XUAT (Top Right)
            if w - 160 < x < w - 20 and 15 < y < 65:
                from tkinter import messagebox
                if messagebox.askyesno("Đăng xuất", "Bạn có chắc chắn muốn đăng xuất?"):
                    self.current_state = STATE_LOGOUT
                    return

            # Row 1 positions
            row1_y = cY - int(100 * (h/600))
            # Row 2 positions
            row2_y = cY + int(0 * (h/600))
            # Row 3 positions
            row3_y = cY + int(100 * (h/600))

            # --- COLUMN 1 ---
            # 1. XEM SERVICE (LIVE)
            if col1_L < x < col1_R and row1_y < y < row1_y + btn_h: 
                self.current_state = STATE_DETECT
                
            # --- ADMIN & COMPANY ACTIONS ---
            if str(self.session_role).lower() in ['admin', 'company']:
                # DANG KY CAM (Col 1, Row 2)
                if col1_L < x < col1_R and row2_y < y < row2_y + btn_h: 
                    self.current_state = STATE_ENROLL_CAM

                # DANG KY FILE (Col 1, Row 3)
                if col1_L < x < col1_R and row3_y < y < row3_y + btn_h: 
                    self.current_state = STATE_ENROLL_UPLOAD

                # --- COLUMN 2 ---
                # CHINH SUA (Col 2, Row 1)
                if col2_L < x < col2_R and row1_y < y < row1_y + btn_h: 
                    self.current_state = STATE_EDIT

                # DANH SACH (Col 2, Row 2)
                elif col2_L < x < col2_R and row2_y < y < row2_y + btn_h: 
                    self.current_state = STATE_LIST
                
                # LICH SU (Col 2, Row 3)
                elif col2_L < x < col2_R and row3_y < y < row3_y + btn_h: 
                    self.current_state = STATE_HISTORY

            # --- ADMIN & COMPANY MANAGEMENT ---
            if str(self.session_role).lower() in ['admin', 'company']:
                # 1. KET NOI HKB (Bottom Left)
                if h - int(84*(h/600)) < y < h - int(20*(h/600)):
                    if cX - int(380*(w/800)) < x < cX - int(140*(w/800)):
                        self.current_state = STATE_HKB_LIST

            # --- SYSTEM-WIDE ACTIONS (Available to all logged-in users) ---
            # 4. CAI DAT (Icon/Small button next to Logout) - TOP LEFT
            if 20 < x < 150 and 15 < y < 65:
                self.current_state = STATE_SETTINGS

            # --- ADMIN ONLY SYSTEM MANAGEMENT ---
            if str(self.session_role).lower() == 'admin':
                # --- ADMIN ONLY BOTTOM BAR ---
                if (h - int(84*(h/600))) < y < (h - int(20*(h/600))):
                    # 2. QUAN LY USER (Bottom Center)
                    if (cX - int(120*(w/800))) < x < (cX + int(120*(w/800))):
                        self.current_state = STATE_CLOUD_USER
                    # 3. QUAN LY CONG TY (Bottom Right)
                    elif (cX + int(140*(w/800))) < x < (cX + int(380*(w/800))): 
                        self.current_state = STATE_COMPANY

    def draw_main_menu(self, w=1280, h=720, service_active=False):
        """Legacy OpenCV menu (Redirection to Dashboard)."""
        # We now use show_main_dashboard instead of draw_main_menu
        return np.zeros((h, w, 3), dtype=np.uint8)

    def show_main_dashboard(self, mongo_db, attendance=None, face_rec=None, camera=None, service_active=False):
        """
        Displays a modern Dashboard using CustomTkinter.
        Returns the selected state.
        """
        import threading
        root = ctk.CTk()
        _apply_icon(root)
        root.title("BITTECH AI - HỆ THỐNG QUẢN LÝ CHẤM CÔNG")
        
        # Window size and position
        w, h = 1010, 660 # Slightly larger for padding
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(screen_w-w)//2}+{(screen_h-h)//2}")
        root.resizable(True, True) # Allow resize
        root.attributes('-topmost', True)

        selected_state = [STATE_MENU]
        role_lower = str(self.session_role).lower()

        def set_state(state):
            """Transitions to a state that needs the main loop (OpenCV windows)."""
            selected_state[0] = state
            self.current_state = state
            root.destroy()

        # --- Popup Actions (Don't close dashboard) ---
        def open_list():
            from src.main import get_target_company
            target_cid = get_target_company(self, mongo_db, allow_selection=True, parent=root)
            if target_cid and attendance:
                # Same logic as management_app.py to get merged list
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
                company_name = mongo_db.get_company_name(target_cid)
                self.show_user_list_ui(all_employees, title=f"Nhân viên - {company_name}", parent=root)

        def open_history():
            from src.main import get_target_company
            target_cid = get_target_company(self, mongo_db, allow_selection=True, parent=root)
            if target_cid:
                company_displayName = mongo_db.get_company_name(target_cid)
                target_date = self.get_date_form(title=f"Lịch sử [{company_displayName}]", ok_button_text="LẤY DỮ LIỆU", parent=root)
                if target_date:
                    logs = mongo_db.get_logs(company_id=target_cid, date=target_date)
                    self.show_attendance_logs_ui(logs, title=f"Lịch sử [{company_displayName}] - {target_date}", 
                                               session_role=self.session_role, 
                                               session_username=self.session_username, 
                                               mongo_db=mongo_db,
                                               target_company=target_cid,
                                               parent=root)

        def open_edit():
            from src.main import handle_edit_logic
            if attendance and face_rec and camera:
                handle_edit_logic(attendance, face_rec, self, camera, parent=root)

        # --- Sidebar ---
        sidebar = ctk.CTkFrame(root, width=220, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        
        logo_label = ctk.CTkLabel(sidebar, text="BITTECH AI", font=("Arial", 24, "bold"), text_color="#1f6aa5")
        logo_label.pack(pady=(30, 40))

        user_info_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        user_info_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(user_info_frame, text=f"Chào, {self.session_username}", font=("Arial", 13, "bold")).pack(anchor="w")
        ctk.CTkLabel(user_info_frame, text=f"Quyền: {role_lower.upper()}", font=("Arial", 11), text_color="gray").pack(anchor="w")

        # Sidebar Buttons
        ctk.CTkButton(sidebar, text="CÀI ĐẶT HỆ THỐNG", 
                     command=lambda: self.show_system_settings_ui(mongo_db, session_username=self.session_username, parent=root), 
                     fg_color="transparent", border_width=1, hover_color="#333333").pack(side="bottom", fill="x", padx=20, pady=10)
        
        ctk.CTkButton(sidebar, text="ĐĂNG XUẤT", command=lambda: set_state(STATE_LOGOUT), 
                     fg_color="#a12c2c", hover_color="#802020").pack(side="bottom", fill="x", padx=20, pady=(10, 0))

        ctk.CTkButton(sidebar, text="THOÁT ỨNG DỤNG", command=lambda: set_state(STATE_EXIT), 
                     fg_color="#444444", hover_color="#222222").pack(side="bottom", fill="x", padx=20, pady=(10, 0))

        # --- Main View ---
        main_view = ctk.CTkFrame(root, corner_radius=0, fg_color="transparent")
        main_view.pack(side="left", fill="both", expand=True, padx=40, pady=30)

        header_label = ctk.CTkLabel(main_view, text="BẢNG ĐIỀU KHIỂN QUẢN TRỊ", font=("Arial", 22, "bold"))
        header_label.pack(pady=(0, 30), anchor="w")

        # Service Status Card
        status_frame = ctk.CTkFrame(main_view, height=80)
        status_frame.pack(fill="x", pady=(0, 30))
        
        dot = ctk.CTkLabel(status_frame, text="", width=15, height=15, corner_radius=8)
        dot.pack(side="left", padx=(20, 10))
        
        status_label = ctk.CTkLabel(status_frame, text="", font=("Arial", 14, "bold"))
        status_label.pack(side="left")

        def update_service_status():
            from src.config import MongoDbConfig
            import time
            nonlocal service_active
            try:
                status_doc = mongo_db.db.system_status.find_one({
                    "type": "camera_service", 
                    "company_id": MongoDbConfig.COMPANY_ID
                })
                if status_doc:
                    last_seen = status_doc.get("last_seen", 0)
                    service_active = (time.time() - last_seen < 15)
                else:
                    service_active = False
            except Exception as e:
                service_active = False
            
            dot_color = "#28a745" if service_active else "#dc3545"
            status_text = "DỊCH VỤ ĐANG HOẠT ĐỘNG" if service_active else "DỊCH VỤ ĐANG TẮT (Vui lòng mở file service_main.exe)"
            
            dot.configure(fg_color=dot_color)
            status_label.configure(text=status_text)
            root.after(3000, update_service_status)

        # Initial update
        update_service_status()

        # --- Button Grid ---
        grid_frame = ctk.CTkFrame(main_view, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True)
        grid_frame.grid_columnconfigure((0, 1), weight=1)
        grid_frame.grid_rowconfigure((0, 1, 2), weight=1)

        # 1. Monitor (All users)
        def watch_live():
            # In DEV mode (not frozen), allow watching live even if service is off
            # because main.py will connect directly to the camera.
            if getattr(sys, 'frozen', False):
                if not service_active:
                    messagebox.showwarning("Dịch Vụ Đang Tắt", 
                                         "Dịch vụ Camera ẩn chưa chạy.\n\nHướng dẫn:\n1. Vui lòng mở file 'service_main.exe' trước khi xem live.")
                    return
                self.use_local_webcam = False
            else:
                self.use_local_webcam = messagebox.askyesno("Nguồn Camera (Debug Mode)", 
                                                            "Bạn có muốn mở Webcam máy tính (Camera 0) thay vì luồng RTSP không?")
            
            selected_state[0] = STATE_DETECT
            self.current_state = STATE_DETECT # Also update self.current_state
            root.destroy()

        btn_monitor = ctk.CTkButton(grid_frame, text="XEM CAMERA TRỰC TIẾP", 
                                   command=watch_live,
                                   height=90, font=("Arial", 15, "bold"),
                                   corner_radius=12, fg_color="#1f6aa5", hover_color="#154c75")
        btn_monitor.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if role_lower in ['admin', 'company']:
            # 2. Edit User
            ctk.CTkButton(grid_frame, text="CHỈNH SỬA THÔNG TIN", 
                         command=open_edit,
                         height=90, font=("Arial", 15, "bold"),
                         corner_radius=12, fg_color="#5D6D7E", hover_color="#34495E").grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

            # 3. Enroll Camera
            ctk.CTkButton(grid_frame, text="ĐĂNG KÝ (CAMERA)", 
                         command=lambda: set_state(STATE_ENROLL_CAM),
                         height=90, font=("Arial", 15, "bold"),
                         corner_radius=12, fg_color="#E67E22", hover_color="#D35400").grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

            # 4. List
            ctk.CTkButton(grid_frame, text="DANH SÁCH NHÂN VIÊN", 
                         command=open_list,
                         height=90, font=("Arial", 15, "bold"),
                         corner_radius=12, fg_color="#8E44AD", hover_color="#732D91").grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

            # 5. Enroll File
            # Dùng set_state → đóng dashboard, main loop sẽ lazy-load face_rec rồi xử lý
            # (Không gọi enroll_by_upload inline vì face_rec=None trong management_app mode)
            def open_enroll_upload():
                set_state(STATE_ENROLL_UPLOAD)

            ctk.CTkButton(grid_frame, text="ĐĂNG KÝ (FILE ẢNH)", 
                         command=open_enroll_upload,
                         height=90, font=("Arial", 15, "bold"),
                         corner_radius=12, fg_color="#16A085", hover_color="#0E6655").grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

            # 6. History
            ctk.CTkButton(grid_frame, text="LỊCH SỬ CHẤM CÔNG", 
                         command=open_history,
                         height=90, font=("Arial", 15, "bold"),
                         corner_radius=12, fg_color="#2E86C1", hover_color="#21618C").grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        # Bottom Buttons
        bottom_frame = ctk.CTkFrame(main_view, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=(20, 0))

        if role_lower in ['admin', 'company']:
            ctk.CTkButton(bottom_frame, text="KẾT NỐI HKB", command=lambda: self.show_hkb_connections_ui(parent=root),
                         height=45, corner_radius=8, fg_color="#28B463", hover_color="#1D8348").pack(side="left", padx=5, expand=True, fill="x")

        if role_lower == 'admin':
            ctk.CTkButton(bottom_frame, text="QUẢN LÝ TÀI KHOẢN", command=lambda: self.show_user_management_ui(mongo_db, parent=root),
                         height=45, corner_radius=8, fg_color="#5DADE2", hover_color="#2E86C1").pack(side="left", padx=5, expand=True, fill="x")
            
            ctk.CTkButton(bottom_frame, text="QUẢN LÝ CÔNG TY", command=lambda: self.show_company_management_ui(mongo_db, parent=root),
                         height=45, corner_radius=8, fg_color="#A569BD", hover_color="#884EA0").pack(side="left", padx=5, expand=True, fill="x")

        root.mainloop()
        return selected_state[0]

    @staticmethod
    def get_user_form(include_upload=False, session_role=None, mongo_db=None, parent=None):
        """
        Modernized enrollment form using CustomTkinter and CTkDateEntry.
        """
        import tkinter as tk
        from tkinter import messagebox, filedialog, ttk
        
        logger.info("Initializing enrollment form UI...")
        
        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title("Form Đăng Ký Người Dùng")
        
        # Center the window
        base_h = 350
        if include_upload: base_h += 120
        if str(session_role).lower() == 'admin': base_h += 80
        
        window_width, window_height = 420, base_h
        try:
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            pos_x = (screen_width // 2) - (window_width // 2)
            pos_y = (screen_height // 2) - (window_height // 2)
            root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        except:
            root.geometry(f"{window_width}x{window_height}")
            
        root.attributes('-topmost', True)
        root.focus_force()

        form_data = {"id": None, "name": None, "bday": None, "files": [], "company_id": None}

        # UI Elements
        ctk.CTkLabel(root, text="ĐĂNG KÝ THÔNG TIN", font=("Arial", 18, "bold"), text_color="#1f6aa5").pack(pady=20)
        
        # Form container
        f = ctk.CTkFrame(root, fg_color="transparent")
        f.pack(fill="both", expand=True, padx=40)

        # ID 
        ctk.CTkLabel(f, text="Mã nhân viên (ID) *:", font=("Arial", 12)).pack(anchor="w")
        entry_id = ctk.CTkEntry(f, width=340, height=35)
        entry_id.pack(pady=(2, 10))

        # Name
        ctk.CTkLabel(f, text="Họ và tên *:", font=("Arial", 12)).pack(anchor="w")
        entry_name = ctk.CTkEntry(f, width=340, height=35)
        entry_name.pack(pady=(2, 10))

        # Birthday using new CTkDateEntry
        ctk.CTkLabel(f, text="Ngày sinh (DD-MM-YYYY):", font=("Arial", 12)).pack(anchor="w")
        date_entry = CTkDateEntry(f, width=340, height=35, placeholder="Chọn ngày sinh")
        date_entry.pack(pady=(2, 10))

        # Company selection for admin
        combo_cid = None
        company_map = {}
        
        if str(session_role).lower() == 'admin' and mongo_db:
            ctk.CTkLabel(f, text="Phân quyền Công ty *:", font=("Arial", 12)).pack(anchor="w")
            try:
                companies = mongo_db.get_all_companies()
                company_map["Admin (admin)"] = "admin"
                company_display_list = ["Admin (admin)"]
                for c in companies:
                    cid = c.get("company_id")
                    cname = c.get("name", cid)
                    display_name = f"{cname} ({cid})"
                    company_map[display_name] = cid
                    company_display_list.append(display_name)
                
                combo_cid = ctk.CTkComboBox(f, values=company_display_list, width=340, height=35)
                combo_cid.set("Admin (admin)")
                combo_cid.pack(pady=(2, 10))
            except Exception as e:
                logger.error(f"UI: Error loading companies: {e}")

        # File upload section
        lbl_file_count = None
        if include_upload:
            ctk.CTkLabel(f, text="Ảnh khuôn mặt *:", font=("Arial", 12)).pack(anchor="w", pady=(10,0))
            
            def on_select_files():
                files = filedialog.askopenfilenames(
                    title="Chọn ảnh khuôn mặt",
                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
                )
                if files:
                    form_data["files"] = list(files)
                    lbl_file_count.configure(text=f"Đã chọn {len(files)} ảnh", text_color="#28a745")
            
            ctk.CTkButton(f, text="Chọn ảnh", command=on_select_files, width=340, height=35, fg_color="#5D6D7E").pack(pady=5)
            lbl_file_count = ctk.CTkLabel(f, text="Chưa chọn ảnh", text_color="gray", font=("Arial", 11))
            lbl_file_count.pack()

        def on_submit():
            u_id = entry_id.get().strip()
            u_name = entry_name.get().strip()
            u_bday = date_entry.get().strip()
            
            if not u_id or not u_name:
                messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ Mã nhân viên và Họ tên!")
                return
            
            if include_upload and not form_data["files"]:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ít nhất một ảnh!")
                return
            
            if str(session_role).lower() == 'admin' and combo_cid:
                display_name = combo_cid.get()
                form_data["company_id"] = company_map.get(display_name, "admin")
            else:
                form_data["company_id"] = None
            
            form_data["id"] = u_id
            form_data["name"] = u_name
            form_data["bday"] = u_bday or "N/A"
            root.destroy()

        # Submit button
        ctk.CTkButton(root, text="XÁC NHẬN ĐĂNG KÝ", command=on_submit, width=250, height=45, 
                     fg_color="#28a745", hover_color="#218838", font=("Arial", 14, "bold")).pack(pady=30)
        
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        if not parent:
            root.mainloop()
        else:
            parent.wait_window(root)
        
        if form_data["id"] is None:
            return None
            
        return form_data["id"], form_data["name"], form_data["bday"], form_data["files"], form_data["company_id"]

    @staticmethod
    def get_id_form():
        """
        Simple dialog to get a User ID.
        """
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        root = tk.Tk()
        _apply_icon(root)
        root.withdraw()
        u_id = simpledialog.askstring("Chỉnh sửa", "Nhập Mã nhân viên cần chỉnh sửa:", parent=root)
        root.destroy()
        return u_id

    @staticmethod
    @staticmethod
    def get_edit_user_form(current_id, current_name, current_bday, session_role=None, parent=None):
        """
        Modernized edit form using CustomTkinter and CTkDateEntry.
        """
        import tkinter as tk
        from tkinter import messagebox

        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title("Chỉnh sửa thông tin")
        root.geometry("450x550")
        root.attributes('-topmost', True)
        root.focus_force()

        result = {
            "name": None, 
            "bday": None, 
            "delete": False,
            "enroll_camera": False,
            "enroll_upload": False
        }
        
        def on_save():
            result["name"] = entry_name.get().strip()
            result["bday"] = date_entry.get().strip()
            if not result["name"]:
                messagebox.showwarning("Cảnh báo", "Họ tên không được để trống!")
                return
            root.destroy()

        def on_delete():
            if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa nhân viên này?"):
                result["delete"] = True
                root.destroy()
        
        def on_enroll_camera():
            result["name"] = entry_name.get().strip()
            result["bday"] = date_entry.get().strip()
            if not result["name"]:
                messagebox.showwarning("Cảnh báo", "Họ tên không được để trống!")
                return
            result["enroll_camera"] = True
            root.destroy()
        
        def on_enroll_upload():
            result["name"] = entry_name.get().strip()
            result["bday"] = date_entry.get().strip()
            if not result["name"]:
                messagebox.showwarning("Cảnh báo", "Họ tên không được để trống!")
                return
            result["enroll_upload"] = True
            root.destroy()

        is_company = (str(session_role).lower() == 'company')

        ctk.CTkLabel(root, text="CHỈNH SỬA NHÂN VIÊN", font=("Arial", 18, "bold"), text_color="#1f6aa5").pack(pady=20)
        
        f = ctk.CTkFrame(root, fg_color="transparent")
        f.pack(fill="both", expand=True, padx=40)

        ctk.CTkLabel(f, text=f"Mã nhân viên: {current_id}", font=("Arial", 13, "bold"), text_color="gray").pack(anchor="w", pady=(0, 15))
        
        ctk.CTkLabel(f, text="Họ và tên:", font=("Arial", 12)).pack(anchor="w")
        entry_name = ctk.CTkEntry(f, width=370, height=35)
        entry_name.insert(0, current_name)
        if is_company: entry_name.configure(state='readonly')
        entry_name.pack(pady=(2, 10))

        ctk.CTkLabel(f, text="Ngày sinh (DD-MM-YYYY):", font=("Arial", 12)).pack(anchor="w")
        date_entry = CTkDateEntry(f, width=370, height=35, initial_date=current_bday)
        if is_company: 
            date_entry.entry.configure(state='readonly')
            date_entry.btn.configure(state='disabled')
        date_entry.pack(pady=(2, 10))

        # Separator
        ctk.CTkLabel(f, text="──────────────────────────────────", text_color="gray").pack(pady=10)
        ctk.CTkLabel(f, text="ĐĂNG KÝ KHUÔN MẶT", font=("Arial", 11, "bold"), text_color="#17a2b8").pack()
        
        # Face enrollment buttons
        face_btn_row = ctk.CTkFrame(f, fg_color="transparent")
        face_btn_row.pack(pady=10, fill="x")
        
        ctk.CTkButton(face_btn_row, text="📷 Camera", command=on_enroll_camera, 
                     width=165, height=35, fg_color="#007bff", hover_color="#0056b3").pack(side="left", padx=(0, 10))
        ctk.CTkButton(face_btn_row, text="📁 File Ảnh", command=on_enroll_upload, 
                     width=165, height=35, fg_color="#6c757d", hover_color="#5a6268").pack(side="left")
        
        # Action buttons
        btn_save = ctk.CTkButton(root, text="LƯU THAY ĐỔI", command=on_save, width=300, height=45, fg_color="#28a745", hover_color="#218838", font=("Arial", 13, "bold"))
        btn_delete = ctk.CTkButton(root, text="XÓA NHÂN VIÊN", command=on_delete, width=300, height=45, fg_color="#dc3545", hover_color="#c82333", font=("Arial", 13, "bold"))
        
        if is_company:
            btn_save.configure(state='disabled', fg_color='#555555')
            btn_delete.configure(state='disabled', fg_color='#555555')
            ctk.CTkLabel(root, text="* Quyền Company không được sửa tên/ngày sinh", text_color="#e74c3c", font=("Arial", 11)).pack()

        btn_save.pack(pady=(20, 10))
        btn_delete.pack(pady=5)
        
        if not parent:
            root.mainloop()
        else:
            parent.wait_window(root)
            
        return result if result["name"] or result["delete"] or result["enroll_camera"] or result["enroll_upload"] else None

    @staticmethod
    def show_user_list_ui(user_list, title="Danh sách nhân viên", parent=None):
        """
        Modernized list of users using CustomTkinter.
        """
        import customtkinter as ctk
        import tkinter as tk
        from tkinter import ttk

        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title(f"Bittech AI - {title}")
        root.geometry("800x600")
        root.attributes('-topmost', True)

        ctk.CTkLabel(root, text=title.upper(), font=("Arial", 20, "bold"), text_color="#1f6aa5").pack(pady=20)

        tree_frame = ctk.CTkFrame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("id", "name", "birthday", "face_status")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        tree.heading("id", text="Mã nhân viên")
        tree.heading("name", text="Họ và tên")
        tree.heading("birthday", text="Ngày sinh")
        tree.heading("face_status", text="Khuôn mặt")
        
        tree.column("id", width=150)
        tree.column("name", width=250)
        tree.column("birthday", width=150)
        tree.column("face_status", width=150, anchor="center")

        for user in user_list:
            u_id = user.get("user_id", "N/A")
            u_name = user.get("user_name") or user.get("name", "Unknown")
            u_bday = user.get("birthday", "N/A")
            has_face = user.get("has_face", True)
            face_status = "✓ Đã đăng ký" if has_face else "⚠ Chưa có"
            tree.insert("", tk.END, values=(u_id, u_name, u_bday, face_status))

        tree.pack(expand=True, fill="both")
        
        ctk.CTkButton(root, text="ĐÓNG CỬA SỔ", command=root.destroy, width=150, height=40).pack(pady=20)

        if not parent:
            root.mainloop()

    @staticmethod
    def pick_company_ui(companies, parent=None):
        """
        Dialog to select a company from a list.
        Returns company_id or None.
        """
        import tkinter as tk
        from tkinter import ttk

        if parent:
            root = tk.Toplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = tk.Tk()
            _apply_icon(root)
            
        root.title("Chọn Công ty")
        root.geometry("400x350")
        root.attributes('-topmost', True)

        tk.Label(root, text="CHỌN CÔNG TY ĐỂ XEM NHÂN VIÊN", font=("Arial", 11, "bold")).pack(pady=10)

        selected = {"id": None}
        
        tree = ttk.Treeview(root, columns=("id", "name"), show="headings")
        tree.heading("id", text="ID")
        tree.heading("name", text="Tên công ty")
        tree.column("id", width=100)
        tree.column("name", width=250)

        # companies list contains company objects
        for c in companies:
            tree.insert("", tk.END, values=(c.get('company_id'), c.get('name')))

        def on_select():
            sel = tree.selection()
            if sel:
                selected["id"] = tree.item(sel[0])['values'][0]
                root.destroy()

        tree.pack(padx=10, pady=10, fill="both", expand=True)
        tk.Button(root, text="XÁC NHẬN", command=on_select, bg="#28a745", fg="white", width=15).pack(pady=10)

        if not parent:
            root.mainloop()
        else:
            parent.wait_window(root)
        return selected["id"]

    @staticmethod
    def pick_user_ui(user_list, parent=None):
        """
        Dialog to select a user from a list.
        Returns user_id or None.
        """
        import tkinter as tk
        from tkinter import ttk

        if parent:
            root = tk.Toplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = tk.Tk()
            _apply_icon(root)
            
        root.title("Chọn Nhân viên")
        root.geometry("600x400")
        root.attributes('-topmost', True)

        tk.Label(root, text="CHỌN NHÂN VIÊN CẦN CHỈNH SỬA", font=("Arial", 11, "bold")).pack(pady=10)

        selected = {"id": None}
        
        tree = ttk.Treeview(root, columns=("id", "name", "bday", "face_status"), show="headings")
        tree.heading("id", text="Mã NV")
        tree.heading("name", text="Họ tên")
        tree.heading("bday", text="Ngày sinh")
        tree.heading("face_status", text="Khuôn mặt")
        
        tree.column("id", width=100)
        tree.column("name", width=200)
        tree.column("bday", width=120)
        tree.column("face_status", width=100, anchor="center")

        for u in user_list:
            u_id = u.get('user_id')
            u_name = u.get('user_name') or u.get('name', 'Unknown')
            u_bday = u.get('birthday', 'N/A')
            has_face = u.get('has_face', True)  # Default True for backward compatibility
            face_status = "✓ Có" if has_face else "⚠ Chưa"
            
            tree.insert("", tk.END, values=(u_id, u_name, u_bday, face_status))

        def on_select():
            sel = tree.selection()
            if sel:
                selected["id"] = tree.item(sel[0])['values'][0]
                root.destroy()

        tree.pack(expand=True, fill="both", padx=10)
        
        btn_frame = tk.Frame(root); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="XÁC NHẬN", command=on_select, bg="green", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="HỦY", command=root.destroy, width=15).pack(side=tk.LEFT, padx=5)

        if not parent:
            root.mainloop()
        else:
            parent.wait_window(root)
        return selected["id"]

    @staticmethod
    def get_date_form(title="Chọn ngày", parent=None, ok_button_text=None):
        """
        Dialog to select a date using CTkDatePicker. Returns YYYY-MM-DD or None.
        """
        picker = CTkDatePicker(parent=parent, title=title, ok_button_text=ok_button_text)
        return picker.get_date()

    @staticmethod
    def pick_system_for_upload(systems, parent=None):
        """
        Dialog to select a system for uploading attendance logs.
        Returns selected system dict or None.
        """
        import tkinter as tk
        from tkinter import ttk

        if parent:
            root = tk.Toplevel(parent)
            _apply_icon(root)
            root.transient(parent)
            root.grab_set()
        else:
            root = tk.Tk()
            _apply_icon(root)
            
        root.title("Chọn Hệ thống Upload")
        root.geometry("600x450")
        root.attributes('-topmost', True)

        tk.Label(root, text="CHỌN HỆ THỐNG ĐỂ UPLOAD CHẤM CÔNG", font=("Arial", 12, "bold")).pack(pady=10)

        selected = {"system": None}
        
        tree = ttk.Treeview(root, columns=("id", "name", "endpoint"), show="headings")
        tree.heading("id", text="System ID")
        tree.heading("name", text="Tên hệ thống")
        tree.heading("endpoint", text="Endpoint")
        tree.column("id", width=150)
        tree.column("name", width=200)
        tree.column("endpoint", width=200)

        for sys in systems:
            tree.insert("", tk.END, values=(
                sys.get('system_id'), 
                sys.get('name'), 
                sys.get('endpoint')
            ))

        def on_select():
            sel = tree.selection()
            if sel:
                values = tree.item(sel[0])['values']
                # Find the full system object
                selected["system"] = next((s for s in systems if s.get('system_id') == values[0]), None)
                root.destroy()

        tree.pack(padx=10, pady=10, fill="both", expand=True)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="XÁC NHẬN", command=on_select, bg="#28a745", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="HỦY", command=root.destroy, width=15).pack(side=tk.LEFT, padx=5)

        if not parent:
            root.mainloop()
        else:
            parent.wait_window(root)
            
        return selected["system"]

    def show_attendance_logs_ui(self, logs, title="Lịch sử điểm danh", session_role=None, session_username="GLOBAL", session_user_id=1, parent=None, mongo_db=None, target_company=None):
        """Modernized UI to view attendance logs using CustomTkinter."""
        from src.services.hkb_service import hkb_service
        
        # Determine initial date from title if possible (Lịch sử ngày YYYY-MM-DD)
        initial_date = ""
        if "ngày " in title:
            initial_date = title.split("ngày ")[-1]
        elif "- " in title:
            initial_date = title.split("- ")[-1]
            
        current_view_date = [initial_date] # Use a list to make it mutable in nested functions
        
        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title(f"Bittech AI - {title}")
        root.geometry("1100x700")
        root.attributes('-topmost', True)

        try:
            display_initial = datetime.strptime(initial_date, "%Y-%m-%d").strftime("%d-%m-%Y") if initial_date else "CHƯA CHỌN"
        except:
            display_initial = initial_date or "CHƯA CHỌN"

        title_label = ctk.CTkLabel(root, text=f"LỊCH SỬ NGÀY {display_initial}", font=("Arial", 22, "bold"), text_color="#1f6aa5")
        title_label.pack(pady=(20, 10))

        # --- Filter Panel ---
        filter_frame = ctk.CTkFrame(root, fg_color="transparent")
        filter_frame.pack(fill="x", padx=20, pady=10)
        
        # Row 1: Date Fetch
        date_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        date_frame.pack(fill="x", pady=0)
        
        ctk.CTkLabel(date_frame, text="Từ ngày:", font=("Arial", 14, "bold")).pack(side="left", padx=(10, 5))
        date_from_entry = CTkDateEntry(date_frame, width=150, height=35, initial_date=initial_date)
        date_from_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(date_frame, text="Đến ngày:", font=("Arial", 14, "bold")).pack(side="left", padx=(20, 5))
        date_to_entry = CTkDateEntry(date_frame, width=150, height=35, initial_date=initial_date)
        date_to_entry.pack(side="left", padx=5)

        def on_filter():
            start_date_str = date_from_entry.get().strip()
            end_date_str = date_to_entry.get().strip()
            if not start_date_str or not end_date_str: return
            
            def convert_date(d_str):
                if "-" in d_str:
                    try:
                        parts = d_str.split("-")
                        if len(parts[0]) == 2: # DD-MM-YYYY
                            return f"{parts[2]}-{parts[1]}-{parts[0]}"
                    except: pass
                return d_str
                
            db_start = convert_date(start_date_str)
            db_end = convert_date(end_date_str)
            
            # Ensure start <= end
            if db_start > db_end:
                db_start, db_end = db_end, db_start
                
            refresh_data_range(db_start, db_end)

        ctk.CTkButton(date_frame, text="LẤY DỮ LIỆU", command=on_filter, width=120, height=35, 
                     fg_color="#1f6aa5", hover_color="#154c75", font=("Arial", 13, "bold")).pack(side="left", padx=10)
                     
        # Row 2: Local Filters
        search_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        search_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkLabel(search_frame, text="Tìm kiếm:", font=("Arial", 13)).pack(side="left", padx=(10, 5))
        search_entry = ctk.CTkEntry(search_frame, width=150, height=30, placeholder_text="Tên hoặc Mã NV...")
        search_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(search_frame, text="Đã tải lên:", font=("Arial", 13)).pack(side="left", padx=(20, 5))
        cloud_var = ctk.StringVar(value="Tất cả")
        cloud_dropdown = ctk.CTkOptionMenu(search_frame, variable=cloud_var, values=["Tất cả", "Rồi (✓)", "Chưa (x)"], width=110, height=30)
        cloud_dropdown.pack(side="left", padx=5)
        
        ctk.CTkLabel(search_frame, text="Trạng thái:", font=("Arial", 13)).pack(side="left", padx=(20, 5))
        status_var = ctk.StringVar(value="Tất cả")
        status_dropdown = ctk.CTkOptionMenu(search_frame, variable=status_var, values=["Tất cả", "IN", "OUT", "FAILED", "SPOOF"], width=110, height=30)
        status_dropdown.pack(side="left", padx=5)
        
        # We need a shared list to hold the currently fetched logs
        all_current_logs = list(logs)
        log_items_batch = {}
        
        def render_tree(logs_to_render):
            for item in tree.get_children():
                tree.delete(item)
            log_items_batch.clear()
            
            for log in logs_to_render:
                l_id = str(log.get('_id', ''))
                ts = log.get('timestamp', 'N/A')
                try:
                    date_part = ts.split(' ')[0]
                    time_part = ts.split(' ')[1]
                except:
                    date_part, time_part = ts, ts
                
                is_up = 1 if log.get('uploaded_to') else 0
                up_str = "✓" if is_up else "x"
                
                tree.insert("", tk.END, values=(
                    l_id, 
                    log.get('user_id', 'N/A'), 
                    log.get('user_name', 'N/A'), 
                    time_part, 
                    date_part, 
                    log.get('status', 'IN'), 
                    up_str
                ))
                log_items_batch[l_id] = log

        def apply_filters(*args):
            keyword = search_entry.get().strip().lower()
            cloud_filter = cloud_var.get()
            status_filter = status_var.get()
            
            filtered = []
            for log in all_current_logs:
                # 1. Search text
                name = str(log.get('user_name', '')).lower()
                uid = str(log.get('user_id', '')).lower()
                if keyword and (keyword not in name and keyword not in uid):
                    continue
                
                # 2. Cloud Filter
                is_up = 1 if log.get('uploaded_to') else 0
                if cloud_filter == "Rồi (✓)" and not is_up:
                    continue
                if cloud_filter == "Chưa (x)" and is_up:
                    continue
                    
                # 3. Status Filter
                log_st = str(log.get('status', 'IN')).upper()
                if status_filter != "Tất cả" and log_st != status_filter:
                    continue
                    
                filtered.append(log)
                
            render_tree(filtered)

        # Bind events for instant filtering without DB hit
        search_entry.bind("<Return>", apply_filters)
        cloud_dropdown.configure(command=apply_filters)
        status_dropdown.configure(command=apply_filters)

        def refresh_data_range(start_date_str, end_date_str):
            if not mongo_db or not target_company:
                return
            new_logs = mongo_db.get_logs(company_id=target_company, start_date=start_date_str, end_date=end_date_str)
            nonlocal all_current_logs
            all_current_logs = list(new_logs)
            apply_filters()
            
            current_view_date[0] = f"{start_date_str} - {end_date_str}"
            try:
                d_start = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%d-%m-%Y")
                d_end = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%d-%m-%Y")
                if d_start == d_end:
                    display_text = f"NGÀY {d_start}"
                else:
                    display_text = f"TỪ {d_start} ĐẾN {d_end}"
            except:
                display_text = f"{start_date_str} ĐẾN {end_date_str}"
                
            title_text = f"LỊCH SỬ {display_text}"
            title_label.configure(text=title_text)
            root.title(f"Bittech AI - {title_text}")

        tree_frame = ctk.CTkFrame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Columns for modern viewing
        columns = ("id", "user_id", "user_name", "time", "date", "status", "uploaded")
        
        # Thêm vertical scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")
        
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended", yscrollcommand=scrollbar.set)
        scrollbar.config(command=tree.yview)
        
        tree.heading("id", text="Mã Log")
        tree.heading("user_id", text="Mã NV")
        tree.heading("user_name", text="Họ và tên")
        tree.heading("time", text="Giờ")
        tree.heading("date", text="Ngày")
        tree.heading("status", text="Trạng thái")
        tree.heading("uploaded", text="Đã tải lên")
        
        tree.column("id", width=80, anchor="center")
        tree.column("user_id", width=100, anchor="center")
        tree.column("user_name", width=200)
        tree.column("time", width=100, anchor="center")
        tree.column("date", width=120, anchor="center")
        tree.column("status", width=100, anchor="center")
        tree.column("uploaded", width=100, anchor="center")
        tree.pack(fill="both", expand=True)

        # Initial Load from logs passed in constructor
        apply_filters()

        def on_view_image():
            sel = tree.selection()
            if not sel: return messagebox.showwarning("!", "Vui lòng chọn ít nhất 1 dòng")
            l_id = str(tree.item(sel[0])['values'][0])
            user_name = tree.item(sel[0])['values'][2]
            
            if not mongo_db: return
            img_bytes = mongo_db.get_log_image(l_id)
            if img_bytes:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    win_img = f"Anh Diem Danh: {user_name}"
                    cv2.namedWindow(win_img, cv2.WINDOW_NORMAL)
                    cv2.imshow(win_img, img)
                    # Force window to top
                    try:
                        cv2.setWindowProperty(win_img, cv2.WND_PROP_TOPMOST, 1)
                    except: pass
                    cv2.waitKey(10) 
                else: messagebox.showerror("Lỗi", "Không thể giải mã hình ảnh")
            else: messagebox.showwarning("Thông báo", "Log này không chứa dữ liệu ảnh")

        def on_batch_upload():
            from src.services.hkb_service import hkb_service
            sel_items = tree.selection()
            if not sel_items: return messagebox.showwarning("!", "Vui lòng chọn các dòng cần upload")
            
            connections = hkb_service.get_connections(user_id=session_user_id, username=session_username)
            connected_systems = [c for c in connections if c.get("client_register") == 1] if connections else []
            if not connected_systems: 
                messagebox.showwarning("!", "Chưa có hệ thống HKB nào được kết nối (Active).")
                return
            
            selected_system = AttendanceUI.pick_system_for_upload(connected_systems, parent=root)
            if not selected_system: return

            upload_logs = []
            log_ids = []
            for item in sel_items:
                l_id = str(tree.item(item)['values'][0])
                log_data = log_items_batch.get(l_id)
                if log_data:
                    log_ids.append(l_id)
                    upload_logs.append({
                        "session_id": l_id,
                        "user_id": log_data.get('user_id'),
                        "user_name": log_data.get('user_name'),
                        "timestamp": log_data.get('timestamp'),
                        "status": log_data.get('status', 'IN'),
                        "image_webp": mongo_db.get_log_image(l_id) if mongo_db else None
                    })

            if not mongo_db: return
            auth_data = mongo_db.auth_services.find_one({"uuid": selected_system["system_id"]})
            if not auth_data: return messagebox.showerror("Lỗi", "Không tìm thấy Auth Key cho hệ thống này")

            root.config(cursor="watch"); root.update()
            res = hkb_service.upload_timekeepers(
                endpoint=selected_system["endpoint"],
                system_id=selected_system["system_id"],
                api_key=auth_data["key"],
                user_id=auth_data.get("user_id", session_user_id),
                attendance_logs=upload_logs
            )
            root.config(cursor="")
            
            if res and res.success:
                mongo_db.mark_logs_uploaded(log_ids, selected_system["system_id"])
                messagebox.showinfo("Hoàn tất", f"Đã tải {len(upload_logs)} dữ liệu lên {selected_system['name']}")
                # REFRESH list after solid upload using the CURRENT date
                if current_view_date[0]:
                    refresh_data(current_view_date[0])
            else: messagebox.showerror("Lỗi Upload", res.message if res else "Không phản hồi từ server")

        def on_double_click(event):
            sel = tree.selection()
            if not sel: return
            l_id = str(tree.item(sel[0])['values'][0])
            log_data = log_items_batch.get(l_id)
            if not log_data: return
            
            detail_win = ctk.CTkToplevel(root)
            detail_win.title(f"Chi tiết Log: {l_id}")
            detail_win.geometry("500x550")
            detail_win.attributes('-topmost', True)
            
            ctk.CTkLabel(detail_win, text="CHI TIẾT ĐIỂM DANH", font=("Arial", 16, "bold")).pack(pady=15)
            
            info_frame = ctk.CTkFrame(detail_win)
            info_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            details = [
                ("Mã Log", l_id),
                ("Nhân viên", f"{log_data.get('user_name')} ({log_data.get('user_id')})"),
                ("Thời gian", log_data.get('timestamp')),
                ("Trạng thái", log_data.get('status')),
                ("Công ty", log_data.get('company_id'))
            ]
            
            for i, (k, v) in enumerate(details):
                ctk.CTkLabel(info_frame, text=f"{k}:", font=("Arial", 12, "bold")).grid(row=i, column=0, padx=10, pady=5, sticky="w")
                ctk.CTkLabel(info_frame, text=str(v)).grid(row=i, column=1, padx=10, pady=5, sticky="w")

            # Upload History
            ctk.CTkLabel(detail_win, text="Lịch sử đồng bộ HKB", font=("Arial", 13, "bold")).pack(pady=(10, 0))
            hist_box = ctk.CTkTextbox(detail_win, height=150)
            hist_box.pack(fill="both", expand=True, padx=20, pady=10)
            
            uploaded_to = log_data.get('uploaded_to', [])
            if not uploaded_to:
                hist_box.insert("1.0", "Chưa được đồng bộ lên hệ thống nào.")
            else:
                hist_text = "Đã đồng bộ thành công tới:\n"
                for sys_id in uploaded_to:
                    hist_text += f"- System UUID: {sys_id}\n"
                hist_box.insert("1.0", hist_text)
            hist_box.configure(state="disabled")

        tree.bind("<Double-1>", on_double_click)

        def on_view_log():
            sel = tree.selection()
            if not sel: return messagebox.showwarning("!", "Vui lòng chọn 1 dòng để xem log")
            if len(sel) > 1: return messagebox.showwarning("!", "Chỉ chọn 1 dòng để xem chi tiết log lỗi")
            
            l_id = str(tree.item(sel[0])['values'][0])
            log_data = log_items_batch.get(l_id)
            if not log_data: return
            
            detail_win = ctk.CTkToplevel(root)
            detail_win.title(f"Chi tiết Log Lỗi: {l_id}")
            detail_win.geometry("600x600")
            detail_win.attributes('-topmost', True)
            
            ctk.CTkLabel(detail_win, text="CHI TIẾT ĐỒNG BỘ ĐẾN HKB", font=("Arial", 16, "bold"), text_color="#e74c3c").pack(pady=15)
            
            hist_box = ctk.CTkTextbox(detail_win)
            hist_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
            
            history = log_data.get('upload_history', [])
            if not history:
                hist_box.insert("1.0", "Không có lịch sử upload hoặc chưa từng đẩy lên máy chủ.")
            else:
                import json
                text_content = ""
                for idx, h in enumerate(reversed(history)):  # Xem log mới nhất trước
                    text_content += f"--- LẦN THỬ THỨ {len(history)-idx} ---\n"
                    text_content += f"Thời gian: {h.get('timestamp')}\n"
                    text_content += f"Hệ thống (UUID): {h.get('system_id')}\n"
                    text_content += f"Trạng thái: {h.get('status')}\n"
                    text_content += f"Lời nhắn: {h.get('message')}\n"
                    if 'error_detail' in h:
                        try:
                            # Thử parse chuỗi JSON string nếu có
                            err_data = h['error_detail']
                            if isinstance(err_data, str):
                                try: err_data = json.loads(err_data)
                                except: pass
                            err_str = json.dumps(err_data, indent=2, ensure_ascii=False)
                            text_content += f"Chi tiết kỹ thuật (Raw):\n{err_str}\n"
                        except:
                            text_content += f"Chi tiết kỹ thuật (Raw): {h['error_detail']}\n"
                    text_content += "\n"
                hist_box.insert("1.0", text_content)
                
            hist_box.configure(state="disabled")

        def on_ui_close():
            try: cv2.destroyAllWindows()
            except: pass
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_ui_close)

        # Action Buttons
        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="XEM ẢNH", command=on_view_image, width=150, fg_color="#f39c12", hover_color="#d35400").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="XEM LỖI (LOG)", command=on_view_log, width=150, fg_color="#e74c3c", hover_color="#c0392b").pack(side="left", padx=10)
        if session_role and str(session_role).lower() in ['admin', 'company']:
            ctk.CTkButton(btn_frame, text="TẢI LÊN HKB", command=on_batch_upload, width=180, fg_color="#28a745", hover_color="#218838").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="ĐÓNG", command=on_ui_close, width=120, fg_color="gray").pack(side="left", padx=10)

        if not parent:
            root.mainloop()

    def show_hkb_connections_ui(self, parent=None):
        """Modernized UI to manage HKB connections using CustomTkinter."""
        from src.services.hkb_service import hkb_service
        from src.config import AuthServiceConfig

        current_user_id = getattr(self, "session_user_id", 1) or 1
        
        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title("Bittech AI - Kết nối AuthService")
        root.geometry("1100x650")
        root.attributes('-topmost', True)

        ctk.CTkLabel(root, text="DANH SÁCH KẾT NỐI HỆ THỐNG", font=("Arial", 22, "bold"), text_color="#1f6aa5").pack(pady=20)

        tree_frame = ctk.CTkFrame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("id", "name", "system_id", "endpoint", "actived", "status")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        tree.heading("id", text="ID")
        tree.heading("name", text="Tên hệ thống")
        tree.heading("system_id", text="System ID")
        tree.heading("endpoint", text="Endpoint")
        tree.heading("actived", text="Hoạt động")
        tree.heading("status", text="Trạng thái")
        
        tree.column("id", width=50, anchor="center")
        tree.column("name", width=250)
        tree.column("system_id", width=200)
        tree.column("endpoint", width=200)
        tree.column("actived", width=100, anchor="center")
        tree.column("status", width=150, anchor="center")
        tree.pack(fill="both", expand=True)

        def refresh_list():
            for item in tree.get_children(): tree.delete(item)
            connections = hkb_service.get_connections(user_id=current_user_id, username=getattr(self, "session_username", "GLOBAL"))
            if connections and isinstance(connections, list):
                for conn in connections:
                    is_registered = conn.get("client_register", 0)
                    status_str = "Đã kết nối" if is_registered == 1 else "Chưa kết nối"
                    tree.insert("", tk.END, values=(conn.get("id", "N/A"), conn.get("name", "N/A"), conn.get("system_id", "N/A"), conn.get("endpoint", "N/A"), "Có" if conn.get("actived") == 1 else "Không", status_str))

        def on_register():
            sel = tree.selection()
            if not sel: return messagebox.showwarning("!", "Vui lòng chọn hệ thống")
            item = tree.item(sel[0])['values']
            root.config(cursor="watch"); root.update()
            res = hkb_service.register_client(system_id=item[2], external_id=current_user_id, description=f"Yêu cầu từ {item[1]}", user_info={"app": "Face Attendance"}, system_connection_id=item[0], system_register=AuthServiceConfig.SYSTEM_ID)
            root.config(cursor=""); refresh_list()
            if res and res.success: messagebox.showinfo("Thành công", "Đã gửi yêu cầu kết nối")
            else: messagebox.showerror("Lỗi", res.message if res else "Thất bại")

        def on_get_employees():
            sel = tree.selection()
            if not sel: return messagebox.showwarning("!", "Vui lòng chọn hệ thống")
            item = tree.item(sel[0])['values']
            if item[5] != "Đã kết nối": return messagebox.showwarning("!", "Chưa được kết nối")
            
            auth_data = mongo_db.auth_services.find_one({"uuid": item[2]})
            if not auth_data: return messagebox.showerror("Lỗi", "Thiếu API Key")
            
            root.config(cursor="watch"); root.update()
            res = hkb_service.get_employees(endpoint=item[3], system_id=item[2], api_key=auth_data["key"], user_id=auth_data.get("user_id", current_user_id))
            root.config(cursor="")
            if res and res.success: self.show_remote_employees_ui(item[1], res.data, item[2], parent=root)
            else: messagebox.showerror("Lỗi", res.message if res else "Thất bại")

        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text="LÀM MỚI", command=refresh_list, width=120).pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="ĐĂNG KÝ KẾT NỐI", command=on_register, width=150, fg_color="#28a745").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="ĐỒNG BỘ NHÂN VIÊN", command=on_get_employees, width=150, fg_color="#17a2b8").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="ĐÓNG", command=root.destroy, width=100, fg_color="gray").pack(side=tk.LEFT, padx=10)
        
        refresh_list()
        if not parent:
            root.mainloop()

    def show_remote_employees_ui(self, system_name, employees, target_company_id, parent=None):
        """Modernized UI to view and save remote employees using CustomTkinter."""
        import tkinter as tk
        from tkinter import ttk, messagebox
        import customtkinter as ctk

        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title(f"Nhân viên từ {system_name}")
        root.geometry("1000x700")
        root.attributes('-topmost', True)

        ctk.CTkLabel(root, text=f"ĐỒNG BỘ NHÂN VIÊN: {system_name.upper()}", font=("Arial", 20, "bold"), text_color="#1f6aa5").pack(pady=20)

        tree_frame = ctk.CTkFrame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("user_id", "name", "bday", "sex", "group")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        tree.heading("user_id", text="Mã NV")
        tree.heading("name", text="Họ và tên")
        tree.heading("bday", text="Ngày sinh")
        tree.heading("sex", text="Giới tính")
        tree.heading("group", text="Phòng ban")
        tree.pack(fill="both", expand=True)

        emp_map = {}
        for emp in employees:
            u_id = emp.get("barcode") or emp.get("user_id") or emp.get("uid") or emp.get("id") or "N/A"
            u_name = emp.get("full_name") or emp.get("name") or emp.get("user_name") or "Unknown"
            u_bday = emp.get("birthday") or "N/A"
            item_id = tree.insert("", tk.END, values=(u_id, u_name, u_bday, emp.get("sex", "-"), emp.get("group_id", "-")))
            emp_map[item_id] = {"id": u_id, "name": u_name, "bday": u_bday, "cid": target_company_id}

        def save_to_db(selected_only=False):
            targets = [emp_map[i] for i in tree.selection()] if selected_only else list(emp_map.values())
            if not targets: return messagebox.showwarning("!", "Không có nhân viên nào")
            if not messagebox.askyesno("Xác nhận", f"Lưu {len(targets)} nhân viên vào hệ thống?"): return
            
            saved = 0
            for t in targets:
                ok, _ = mongo_db.save_employee(t["id"], t["name"], t["bday"], t["cid"])
                if ok: saved += 1
            
            messagebox.showinfo("Kết quả", f"Đã lưu thành công {saved}/{len(targets)} nhân viên.")
            if saved > 0: root.destroy()

        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text="LƯU ĐÃ CHỌN", command=lambda: save_to_db(True), width=150, fg_color="#28a745").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="LƯU TẤT CẢ", command=lambda: save_to_db(False), width=150, fg_color="#17a2b8").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="HỦY", command=root.destroy, width=100, fg_color="gray").pack(side=tk.LEFT, padx=10)

        if not parent:
            root.mainloop()

    def show_company_management_ui(self, mongo_db, parent=None):
        """Modernized UI for Company Management (Admin Only) using CustomTkinter."""

        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title("Bittech AI - Quản lý công ty")
        root.geometry("800x600")
        root.attributes('-topmost', True)

        ctk.CTkLabel(root, text="DANH SÁCH CÔNG TY TRÊN HỆ THỐNG", font=("Arial", 20, "bold"), text_color="#1f6aa5").pack(pady=20)

        tree_frame = ctk.CTkFrame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("id", "name", "desc")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        tree.heading("id", text="Mã công ty (ID)")
        tree.heading("name", text="Tên công ty")
        tree.heading("desc", text="Mô tả")
        
        tree.column("id", width=150)
        tree.column("name", width=250)
        tree.column("desc", width=250)
        tree.pack(fill="both", expand=True)

        def refresh():
            for item in tree.get_children(): tree.delete(item)
            for c in mongo_db.get_all_companies():
                tree.insert("", tk.END, values=(c.get('company_id'), c.get('name'), c.get('description')))

        def on_add():
            add_win = ctk.CTkToplevel(root)
            add_win.title("Thêm công ty mới")
            add_win.geometry("400x400")
            add_win.attributes('-topmost', True)
            
            ctk.CTkLabel(add_win, text="THÔNG TIN CÔNG TY", font=("Arial", 16, "bold")).pack(pady=20)
            
            ctk.CTkLabel(add_win, text="Mã Công ty (ID):").pack(anchor="w", padx=40)
            e_id = ctk.CTkEntry(add_win, width=320)
            e_id.pack(pady=5)
            
            ctk.CTkLabel(add_win, text="Tên Công ty:").pack(anchor="w", padx=40)
            e_name = ctk.CTkEntry(add_win, width=320)
            e_name.pack(pady=5)
            
            ctk.CTkLabel(add_win, text="Mô tả:").pack(anchor="w", padx=40)
            e_desc = ctk.CTkEntry(add_win, width=320)
            e_desc.pack(pady=5)
            
            def submit():
                cid, name, desc = e_id.get().strip(), e_name.get().strip(), e_desc.get().strip()
                if not cid or not name: return messagebox.showwarning("!", "Vui lòng nhập đầy đủ ID và Tên công ty")
                success, msg = mongo_db.create_company(cid, name, desc)
                if success:
                    messagebox.showinfo("Thành công", f"Đã thêm công ty {name} thành công")
                    add_win.destroy()
                    refresh()
                else: messagebox.showerror("Lỗi", msg)
            
            ctk.CTkButton(add_win, text="LƯU CÔNG TY", command=submit, height=40, font=("Arial", 14, "bold")).pack(pady=30)

        def on_delete():
            sel = tree.selection()
            if not sel: return
            cid = tree.item(sel[0])['values'][0]
            if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa công ty {cid}?\nTất cả nhân viên thuộc công ty này sẽ bị mất liên kết."):
                if mongo_db.delete_company(cid):
                    messagebox.showinfo("OK", "Đã xóa công ty thành công")
                    refresh()
                else: messagebox.showerror("Lỗi", "Không thể xóa công ty này")

        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="LÀM MỚI", command=refresh, width=120).pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="THÊM MỚI", command=on_add, width=120, fg_color="#28a745", hover_color="#218838").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="XÓA CÔNG TY", command=on_delete, width=120, fg_color="#dc3545", hover_color="#c82333").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="ĐÓNG", command=root.destroy, width=100, fg_color="gray").pack(side=tk.LEFT, padx=10)
        
        refresh()
        if not parent:
            root.mainloop()

    def show_user_management_ui(self, mongo_db, parent=None):
        """Modernized UI to manage system users (Admin Only) using CustomTkinter."""

        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title("Bittech AI - Quản lý tài khoản")
        root.geometry("800x600")
        root.attributes('-topmost', True)

        ctk.CTkLabel(root, text="DANH SÁCH TÀI KHOẢN HỆ THỐNG", font=("Arial", 20, "bold"), text_color="#1f6aa5").pack(pady=20)

        # Treeview (still using standard ttk for tabular data, but wrapped in custom frame)
        tree_frame = ctk.CTkFrame(root)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        columns = ("user", "role", "company")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        tree.heading("user", text="Tên đăng nhập")
        tree.heading("role", text="Quyền")
        tree.heading("company", text="Mã công ty")
        tree.pack(fill="both", expand=True)

        def refresh():
            for item in tree.get_children(): tree.delete(item)
            users = mongo_db.db.users.find()
            for u in users:
                tree.insert("", tk.END, values=(u.get("username"), u.get("role"), u.get("company_id")))

        def on_add():
            add_win = ctk.CTkToplevel(root)
            add_win.title("Thêm tài khoản mới")
            add_win.geometry("400x500")
            add_win.attributes('-topmost', True)
            
            ctk.CTkLabel(add_win, text="THÊM TÀI KHOẢN", font=("Arial", 16, "bold")).pack(pady=20)
            
            ctk.CTkLabel(add_win, text="Username:").pack(anchor="w", padx=40)
            e_user = ctk.CTkEntry(add_win, width=320)
            e_user.pack(pady=5)
            
            ctk.CTkLabel(add_win, text="Password:").pack(anchor="w", padx=40)
            e_pwd = ctk.CTkEntry(add_win, width=320, show="*")
            e_pwd.pack(pady=5)
            
            ctk.CTkLabel(add_win, text="Quyền hạn:").pack(anchor="w", padx=40)
            e_role = ctk.CTkComboBox(add_win, values=["admin", "company", "staff"], width=320)
            e_role.pack(pady=5)
            
            ctk.CTkLabel(add_win, text="Gán cho công ty:").pack(anchor="w", padx=40)
            # Fetch company list
            all_companies = mongo_db.db.companies.find()
            company_map = {"Admin (Cổng Tổng)": "admin"}
            company_display_list = ["Admin (Cổng Tổng)"]
            for c in all_companies:
                cid = c.get("company_id")
                cname = c.get("name", cid)
                display_text = f"{cname} ({cid})"
                company_map[display_text] = cid
                company_display_list.append(display_text)
            
            e_cid = ctk.CTkComboBox(add_win, values=company_display_list, width=320)
            e_cid.set("Admin (Cổng Tổng)"); e_cid.pack(pady=5)
            
            def submit():
                u, p, r = e_user.get(), e_pwd.get(), e_role.get()
                display_cid = e_cid.get()
                cid = company_map.get(display_cid, "admin")
                
                if not u or not p or not cid: return messagebox.showwarning("!", "Vui lòng nhập đầy đủ thông tin")
                
                success, msg = mongo_db.create_user(u, p, r, cid)
                if success:
                    messagebox.showinfo("Thành công", f"Đã tạo tài khoản {u} thành công!")
                    add_win.destroy()
                    refresh()
                else:
                    messagebox.showerror("Lỗi", msg)

            ctk.CTkButton(add_win, text="LƯU TÀI KHOẢN", command=submit, height=40, font=("Arial", 14, "bold")).pack(pady=30)

        def on_delete():
            sel = tree.selection()
            if not sel: return
            user = tree.item(sel[0])['values'][0]
            if user == "admin": return messagebox.showwarning("!", "Không thể xóa Super Admin")
            if messagebox.askyesno("Xác nhận", f"Xóa tài khoản {user}?"):
                if mongo_db.delete_user(user):
                    messagebox.showinfo("OK", "Đã xóa tài khoản")
                    refresh()
                else: messagebox.showerror("Lỗi", "Không thể xóa")

        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="LÀM MỚI", command=refresh, width=120).pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="THÊM MỚI", command=on_add, width=120, fg_color="#28a745", hover_color="#218838").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="XÓA TÀI KHOẢN", command=on_delete, width=120, fg_color="#dc3545", hover_color="#c82333").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="ĐÓNG", command=root.destroy, width=100, fg_color="gray").pack(side=tk.LEFT, padx=10)
        
        refresh()
        if not parent:
            root.mainloop()

    def show_login_dialog(self):
        """
        Shows a modern login dialog for admin authentication using CustomTkinter.
        """
        # Close any existing OpenCV windows to avoid overlap issues if needed
        # cv2.destroyAllWindows() 
        
        login_root = ctk.CTk()
        _apply_icon(login_root)
        login_root.title("Bittech AI - Đăng nhập")
        window_width, window_height = 400, 450
        
        # Center the window
        screen_width = login_root.winfo_screenwidth()
        screen_height = login_root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        login_root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        login_root.attributes('-topmost', True)
        login_root.resizable(False, False)
        
        login_status = {"authenticated": False}

        def attempt_login(event=None):
            user = entry_user.get()
            pwd = entry_pwd.get()
            
            # Authenticate using Cloud MongoDB
            auth_info = mongo_db.verify_login(user, pwd)
            
            if auth_info:
                login_status["authenticated"] = True
                self.is_admin_logged_in = True 
                self.session_role = auth_info["role"]
                self.session_company_id = auth_info["company_id"]
                self.session_username = auth_info["username"]
                self.session_user_id = auth_info.get("user_id", 1)
                login_root.destroy()
            else:
                messagebox.showerror("Lỗi đăng nhập", "Sai tài khoản hoặc mật khẩu hệ thống Cloud!")

        # UI Construction
        main_frame = ctk.CTkFrame(login_root)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(main_frame, text="HỆ THỐNG AI CHẤM CÔNG", font=("Arial", 18, "bold"), text_color="#1f6aa5").pack(pady=(20, 30))
        
        ctk.CTkLabel(main_frame, text="Tên đăng nhập", font=("Arial", 12)).pack(anchor="w", padx=30)
        entry_user = ctk.CTkEntry(main_frame, width=280, height=35, placeholder_text="Nhập username...")
        entry_user.pack(pady=(5, 15))
        
        ctk.CTkLabel(main_frame, text="Mật khẩu", font=("Arial", 12)).pack(anchor="w", padx=30)
        entry_pwd = ctk.CTkEntry(main_frame, width=280, height=35, placeholder_text="Nhập mật khẩu...", show="*")
        entry_pwd.pack(pady=(5, 30))
        
        btn_login = ctk.CTkButton(main_frame, text="ĐĂNG NHẬP", command=attempt_login, width=280, height=40, font=("Arial", 14, "bold"))
        btn_login.pack(pady=10)
        
        def on_exit_app():
            login_status["authenticated"] = "EXIT"
            login_root.destroy()

        ctk.CTkButton(main_frame, text="THOÁT", command=on_exit_app, width=150, fg_color="gray").pack(pady=5)
        
        # Support Enter key
        login_root.bind('<Return>', attempt_login)

        login_root.mainloop()
        return login_status["authenticated"]

    @staticmethod
    def show_system_settings_ui(mongo_db, session_username="GLOBAL", parent=None):
        """Modernized UI to manage settings like Group Keys and Camera using CustomTkinter."""
        from src.config import MongoDbConfig
        
        # Use COMPANY_ID to ensure that settings are physical-machine scoped in DB instead of user-session scoped.
        config_scope = MongoDbConfig.COMPANY_ID
        
        if parent:
            root = ctk.CTkToplevel(parent)
            root.transient(parent)
            _apply_icon(root)
        else:
            root = ctk.CTk()
            _apply_icon(root)
            
        root.title("Bittech AI - Cài đặt hệ thống")
        root.geometry("650x600")
        root.attributes('-topmost', True)
        root.resizable(False, False)

        ctk.CTkLabel(root, text="CẤU HÌNH HỆ THỐNG", font=("Arial", 20, "bold"), text_color="#1f6aa5").pack(pady=20)

        main_frame = ctk.CTkFrame(root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # --- A. CAMERA CONFIG ---
        cam_group = ctk.CTkFrame(main_frame, fg_color="transparent")
        cam_group.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(cam_group, text="CẤU HÌNH CAMERA (RTSP)", font=("Arial", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        # IP & Port row
        ctk.CTkLabel(cam_group, text="IP Camera:").grid(row=1, column=0, sticky="w", pady=5)
        e_ip = ctk.CTkEntry(cam_group, width=180, placeholder_text="192.168.1.100")
        e_ip.grid(row=1, column=1, padx=5, sticky="w")
        e_ip.insert(0, mongo_db.get_setting("camera_ip", "192.168.1.1", username=config_scope))
        
        ctk.CTkLabel(cam_group, text="Port:").grid(row=1, column=2, sticky="w", pady=5, padx=(10, 0))
        e_port = ctk.CTkEntry(cam_group, width=80, placeholder_text="554")
        e_port.grid(row=1, column=3, padx=5, sticky="w")
        e_port.insert(0, mongo_db.get_setting("camera_port", "554", username=config_scope))
        
        # User & Pass row
        ctk.CTkLabel(cam_group, text="Tài khoản:").grid(row=2, column=0, sticky="w", pady=5)
        e_user = ctk.CTkEntry(cam_group, width=180, placeholder_text="admin")
        e_user.grid(row=2, column=1, padx=5, sticky="w")
        e_user.insert(0, mongo_db.get_setting("camera_user", "admin", username=config_scope))
        
        ctk.CTkLabel(cam_group, text="Mật khẩu:").grid(row=2, column=2, sticky="w", pady=5, padx=(10, 0))
        e_pass = ctk.CTkEntry(cam_group, width=180, placeholder_text="password", show="*")
        e_pass.grid(row=2, column=3, padx=5, sticky="w")
        e_pass.insert(0, mongo_db.get_setting("camera_pass", "password", username=config_scope))

        # Recognition & Cooldown Settings
        ctk.CTkLabel(cam_group, text="NHẬN DIỆN & KHÓA", font=("Arial", 13, "bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(15, 10))
        
        ctk.CTkLabel(cam_group, text="Thời gian khóa (phút):").grid(row=4, column=0, sticky="w", pady=5)
        e_cooldown = ctk.CTkEntry(cam_group, width=180, placeholder_text="60")
        e_cooldown.grid(row=4, column=1, padx=5, sticky="w")
        
        # Get current cooldown (stored in SECONDS, display in MINUTES)
        from src.config import RecognitionConfig
        current_cooldown_sec = int(mongo_db.get_setting("detection_cooldown", str(RecognitionConfig.COOLDOWN_SECONDS), username=config_scope))
        e_cooldown.insert(0, str(current_cooldown_sec // 60))

        # Anti-spoofing toggle
        anti_spoof_var = ctk.StringVar()
        is_anti_spoof_enabled = str(mongo_db.get_setting("anti_spoofing_enabled", str(RecognitionConfig.ANTI_SPOOFING_ENABLED), username=config_scope)).lower() == "true"
        anti_spoof_var.set("True" if is_anti_spoof_enabled else "False")
        chk_anti_spoof = ctk.CTkCheckBox(cam_group, text="Bật chống giả mạo (Anti-Spoofing)", variable=anti_spoof_var, onvalue="True", offvalue="False", font=("Arial", 12))
        chk_anti_spoof.grid(row=4, column=2, columnspan=2, padx=(10, 0), pady=5, sticky="w")
        
        # ROI config
        ctk.CTkLabel(cam_group, text="Vùng quét thẻ (ROI):").grid(row=5, column=0, sticky="w", pady=5)
        e_roi = ctk.CTkEntry(cam_group, width=180, placeholder_text="Mặc định (Toàn màn hình)")
        e_roi.grid(row=5, column=1, padx=5, sticky="w")
        e_roi.insert(0, mongo_db.get_setting("camera_roi", "", username=config_scope))
        
        def pick_roi():
            import cv2
            from src.camera.rtsp_camera import RTSPCamera
            
            # Use current text box values to connect
            ip = e_ip.get().strip()
            port = e_port.get().strip()
            user = e_user.get().strip()
            pwd = e_pass.get().strip()
            test_url = f"rtsp://{user}:{pwd}@{ip}:{port}/ch1/main"
            if ip.isdigit(): test_url = ip
            
            try:
                cam = RTSPCamera(rtsp_url=str(test_url))
                if not cam.connect():
                    messagebox.showerror("Lỗi", "Không thể kết nối Camera để vẽ ROI!")
                    return
                print("Dang doc hinh anh de ve ROI live...")
                
                # Biến lưu trạng thái vẽ
                drawing = False
                roi_rect = [0, 0, 0, 0] # x, y, w, h
                drag_start_pt = None
                win_name_roi = "Ve ROI (Keo chuot trai de ve, ENTER/SPACE lap xong, C tao lai, ESC huy)"
                
                def draw_roi_event(event, x, y, flags, param):
                    nonlocal drawing, roi_rect, drag_start_pt
                    if event == cv2.EVENT_LBUTTONDOWN:
                        drawing = True
                        drag_start_pt = (x, y)
                        roi_rect = [x, y, 0, 0]
                    elif event == cv2.EVENT_MOUSEMOVE:
                        if drawing:
                            roi_rect[2] = x - drag_start_pt[0]
                            roi_rect[3] = y - drag_start_pt[1]
                    elif event == cv2.EVENT_LBUTTONUP:
                        drawing = False
                        roi_rect[2] = x - drag_start_pt[0]
                        roi_rect[3] = y - drag_start_pt[1]
                        
                        # Normalize negative width/height
                        if roi_rect[2] < 0:
                            roi_rect[0] += roi_rect[2]
                            roi_rect[2] = abs(roi_rect[2])
                        if roi_rect[3] < 0:
                            roi_rect[1] += roi_rect[3]
                            roi_rect[3] = abs(roi_rect[3])
                
                cv2.namedWindow(win_name_roi, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win_name_roi, cv2.WND_PROP_TOPMOST, 1)
                cv2.setMouseCallback(win_name_roi, draw_roi_event)
                
                selected = False
                while True:
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        continue
                    
                    display = frame.copy()
                    
                    # Draw current rect
                    x, y, w, h = roi_rect
                    if w > 0 and h > 0:
                        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    cv2.putText(display, "[Keo chuot de ban] [ENTER/SPACE] Xong  [C] Xoa Ve Lai  [ESC] Huy", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                    cv2.imshow(win_name_roi, display)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key in [13, 32]: # ENTER or SPACE
                        if w > 0 and h > 0:
                            selected = True
                        break
                    elif key == 27: # ESC
                        break
                    elif key == ord('c'):
                        roi_rect = [0, 0, 0, 0]
                
                cam.disconnect()
                cv2.destroyWindow(win_name_roi)
                
                if selected:
                    x, y, w, h = roi_rect
                    x2, y2 = x + w, y + h
                    roi_string = f"{x},{y},{x2},{y2}"
                    e_roi.delete(0, 'end')
                    e_roi.insert(0, roi_string)
                    messagebox.showinfo("ROI", f"Đã nhận vùng quét: ({x},{y}) đến ({x2},{y2})")
                else:
                    messagebox.showinfo("ROI", "Đã hủy thao tác vẽ khu vực.")
            except Exception as e:
                messagebox.showerror("Lỗi OpenCV", f"Không thể vẽ ROI: {e}")
                
        roi_btn = ctk.CTkButton(cam_group, text="Vẽ khung Camera", command=pick_roi, width=120, fg_color="#2196F3")
        roi_btn.grid(row=5, column=2, columnspan=2, padx=(10, 0), pady=5, sticky="w")

        # --- B. GROUP KEYS ---
        keys_group = ctk.CTkFrame(main_frame, fg_color="transparent")
        keys_group.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(keys_group, text="GROUP KEYS (Phân cách bởi dấu phẩy)", font=("Arial", 13, "bold")).pack(anchor="w", pady=(0, 5))
        
        current_keys = mongo_db.get_setting("group_keys", "", username=config_scope)
        text_keys = ctk.CTkTextbox(keys_group, height=120, font=("Consolas", 12))
        text_keys.pack(fill="both", expand=True, pady=5)
        text_keys.insert("1.0", current_keys)

        def save_settings():
            new_keys = text_keys.get("1.0", "end-1c").strip()
            ip = e_ip.get().strip()
            port = e_port.get().strip()
            user = e_user.get().strip()
            pwd = e_pass.get().strip()
            cooldown_min = e_cooldown.get().strip()
            roi_val = e_roi.get().strip()
            
            success = True
            success &= mongo_db.set_setting("group_keys", new_keys, username=config_scope)
            success &= mongo_db.set_setting("camera_ip", ip, username=config_scope)
            success &= mongo_db.set_setting("camera_port", port, username=config_scope)
            success &= mongo_db.set_setting("camera_user", user, username=config_scope)
            success &= mongo_db.set_setting("camera_pass", pwd, username=config_scope)
            
            # Save cooldown (convert MINUTES to SECONDS for backend)
            try:
                cooldown_sec = int(cooldown_min) * 60
                success &= mongo_db.set_setting("detection_cooldown", str(cooldown_sec), username=config_scope)
                # Update global config immediately
                from src.config import RecognitionConfig
                RecognitionConfig.COOLDOWN_SECONDS = cooldown_sec
            except ValueError:
                messagebox.showerror("Lỗi", "Thời gian khóa phải là một con số!")
                return
                
            # Save anti-spoofing config
            is_anti_spoof = anti_spoof_var.get() == "True"
            success &= mongo_db.set_setting("anti_spoofing_enabled", str(is_anti_spoof), username=config_scope)
            RecognitionConfig.ANTI_SPOOFING_ENABLED = is_anti_spoof
            
            # Save ROI
            success &= mongo_db.set_setting("camera_roi", roi_val, username=config_scope)
            
            if success:
                # Force apply camera configs immediately
                from src.config import CameraConfig
                CameraConfig.IP = ip
                CameraConfig.PORT = int(port)
                CameraConfig.USER = user
                CameraConfig.PASS = pwd
                CameraConfig.RTSP_URL = f"rtsp://{user}:{pwd}@{ip}:{port}/ch1/main"
                
                try:
                    if roi_val and len(roi_val.split(',')) == 4:
                        CameraConfig.ROI = tuple(map(int, roi_val.split(',')))
                    else:
                        CameraConfig.ROI = None
                except:
                    CameraConfig.ROI = None

                # Attempt to restart background service if needed, but for now we apply the config
                messagebox.showinfo("Thành công", "Đã lưu cài đặt hệ thống và cập nhật cấu hình trực tiếp (Live).\nNếu dùng dịch vụ ngầm (Service), Camera sẽ tự nhận luồng mới ở lần kết nối lại tiếp theo.")
                root.destroy()
            else:
                messagebox.showerror("Lỗi", "Không thể lưu cài đặt!")

        # Action Buttons
        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=20, side=tk.BOTTOM)
        
        def restart_service():
            import subprocess
            import os
            try:
                # Tìm và kill tiến trình service_main.exe
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/F", "/IM", "service_main.exe"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Thử mở lại service (nếu ở cùng thư mục)
                    service_path = os.path.join(os.getcwd(), "service_main.exe")
                    if os.path.exists(service_path):
                        subprocess.Popen([service_path])
                        messagebox.showinfo("Thành công", "Đã gửi lệnh khởi động lại Camera Service!")
                    else:
                        messagebox.showinfo("Thông báo", "Đã đóng Camera Service cũ. Vui lòng mở lại file 'service_main.exe' thủ công nếu cần tính năng ngầm.")
                else:
                    messagebox.showinfo("Thông báo", "Chức năng này hiện chỉ hỗ trợ trên Windows.")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể khởi động lại service: {e}")

        ctk.CTkButton(btn_frame, text="KHỞI ĐỘNG LẠI SERVICE", command=restart_service, width=180, height=40, font=("Arial", 11, "bold"), fg_color="#e67e22", hover_color="#d35400").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="LƯU CÀI ĐẶT", command=save_settings, width=150, height=40, font=("Arial", 13, "bold"), fg_color="#28a745", hover_color="#218838").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="ĐÓNG", command=root.destroy, width=100, height=40, fg_color="gray", hover_color="#555555").pack(side=tk.LEFT, padx=10)

        if not parent:
            root.mainloop()

    @staticmethod
    def draw_status_bar(frame, fps, processing_time):
        """Draw FPS and AI stats."""
        cv2.putText(frame, f"FPS: {fps} | AI: {processing_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Nhan 'M' de ve Menu", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return frame
