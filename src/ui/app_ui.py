import cv2
import numpy as np
from loguru import logger
from src.attendance.mongodb_mgr import mongo_db
from tkinter import messagebox

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

            # --- COLUMN 1 ACTIONS ---
            # 1. XEM SERVICE (LIVE)
            row1_y = cY - int(120 * (h/600))
            row1_5_y = cY - int(40 * (h/600))
            row2_y = cY + int(40 * (h/600))
            row3_y = cY + int(120 * (h/600))

            if col1_L < x < col1_R and row1_y < y < row1_y + btn_h: 
                self.current_state = STATE_DETECT
                
            # --- ADMIN & COMPANY RECOGNITION ACTIONS ---
            if str(self.session_role).lower() in ['admin', 'company']:
                # DANG KY CAM (Col 1, Row 2)
                if col1_L < x < col1_R and row2_y < y < row2_y + btn_h: 
                    self.current_state = STATE_ENROLL_CAM

                # DANG KY FILE (Col 1, Row 3)
                if col1_L < x < col1_R and row3_y < y < row3_y + btn_h: 
                    self.current_state = STATE_ENROLL_UPLOAD

                # --- COLUMN 2 ACTIONS ---
                # CHINH SUA (Align with Row 1)
                if col2_L < x < col2_R and row1_y < y < row1_y + btn_h: 
                    self.current_state = STATE_EDIT

                # DANH SACH (Align with Row 1.5)
                elif col2_L < x < col2_R and row1_5_y < y < row1_5_y + btn_h: 
                    self.current_state = STATE_LIST
                
                # LICH SU (Align with Row 2)
                elif col2_L < x < col2_R and row2_y < y < row2_y + btn_h: 
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
                if h - int(84*(h/600)) < y < h - int(20*(h/600)):
                    # 2. QUAN LY USER (Bottom Center)
                    if cX - int(120*(w/800)) < x < cX + int(120*(w/800)):
                        self.current_state = STATE_CLOUD_USER
                    # 3. QUAN LY CONG TY (Bottom Right)
                    elif cX + int(140*(w/800)) < x < cX + int(380*(w/800)): 
                        self.current_state = STATE_COMPANY

    def draw_main_menu(self, w=1280, h=720, service_active=False):
        """Draw a professional menu responsive to window size."""
        self.service_active = service_active # Store state for click handling
        self.last_w, self.last_h = w, h
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(frame, (0, 0), (w, h), (40, 40, 40), -1) 
        
        cX, cY = w // 2, h // 2
        
        # Title
        title_font_scale = w / 800 * 1.2
        cv2.putText(frame, "HE THONG DIEM DANH AI", (cX - int(240 * (w/800)), cY - int(150 * (h/600))),
                    cv2.FONT_HERSHEY_DUPLEX, title_font_scale, (255, 255, 255), 2)

        # Draw Columns Layout
        btn_w, btn_h = int(300 * (w/800)), int(60 * (h/600))
        gap_x = int(10 * (w/800))
        gap_y = int(20 * (h/600))
        role_lower = str(self.session_role).lower()
        
        col1_x = cX - btn_w - gap_x
        col2_x = cX + gap_x

        # Define Rows (Shared for both columns)
        row1_y = cY - int(120 * (h/600))
        row1_5_y = cY - int(40 * (h/600))
        row2_y = cY + int(40 * (h/600))
        row3_y = cY + int(120 * (h/600))

        # --- Column 1 ---
        # Button 1: Monitor Service (All users)
        btn_color = (40, 180, 40) if service_active else (60, 60, 60)
        cv2.rectangle(frame, (col1_x, row1_y), (col1_x + btn_w, row1_y + btn_h), btn_color, -1)
        cv2.putText(frame, "XEM SERVICE (LIVE)", (col1_x + int(45 * (w/800)), row1_y + int(38 * (h/600))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * (w/800), (255, 255, 255), 2)
        
        # Status Dot for Service
        dot_color = (0, 255, 0) if service_active else (0, 0, 255)
        cv2.circle(frame, (col1_x + int(20 * (w/800)), row1_y + int(30 * (h/600))), 8, dot_color, -1)
        
        # Enrollment Buttons (Available to Admin and Company Managers)
        if role_lower in ['admin', 'company']:
            # Button 2: Enroll Camera
            cv2.rectangle(frame, (col1_x, row2_y), (col1_x + btn_w, row2_y + btn_h), (200, 120, 0), -1)
            cv2.putText(frame, "DANG KY (CAM)", (col1_x + int(50 * (w/800)), row2_y + int(38 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)
            
            # Button 3: Enroll Upload
            cv2.rectangle(frame, (col1_x, row3_y), (col1_x + btn_w, row3_y + btn_h), (0, 100, 200), -1)
            cv2.putText(frame, "DANG KY (FILE)", (col1_x + int(50 * (w/800)), row3_y + int(38 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)
        
        # Management Buttons (Admin & Company Only)
        if role_lower in ['admin', 'company']:
            # Button 4: Edit (Align with Row 1)
            cv2.rectangle(frame, (col2_x, row1_y), (col2_x + btn_w, row1_y + btn_h), (100, 100, 100), -1)
            cv2.putText(frame, "CHINH SUA", (col2_x + int(75 * (w/800)), row1_y + int(38 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)

            # Button 5: List (Align with Row 1.5)
            cv2.rectangle(frame, (col2_x, row1_5_y), (col2_x + btn_w, row1_5_y + btn_h), (150, 50, 150), -1)
            cv2.putText(frame, "DANH SACH", (col2_x + int(75 * (w/800)), row1_5_y + int(38 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)

            # Button 6: History (Align with Row 2)
            cv2.rectangle(frame, (col2_x, row2_y), (col2_x + btn_w, row2_y + btn_h), (100, 50, 0), -1)
            cv2.putText(frame, "LICH SU", (col2_x + int(90 * (w/800)), row2_y + int(38 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)
        
        # --- Instruction Table (Compact) ---
        table_x, table_y = int(30 * (w/800)), int(480 * (h/600))
        table_width, table_height = int(200 * (w/800)), int(90 * (h/600))
        cv2.rectangle(frame, (table_x, table_y), (table_x + table_width, table_y + table_height), (60, 60, 60), -1)
        cv2.rectangle(frame, (table_x, table_y), (table_x + table_width, table_y + table_height), (100, 100, 100), 1)
        
        instruction_font_scale = 0.5 * (w/800)
        instruction_line_height = int(15 * (h/600))
        cv2.putText(frame, "PHIM TAT:", (table_x + int(10 * (w/800)), table_y + int(20 * (h/600))),
                    cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, (255, 255, 0), 1)
        
        instructions = ["Q: Thoat", "M: Menu", "S: Chup anh", "C: Huy bỏ"]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (table_x + int(10 * (w/800)), table_y + int(40 * (h/600)) + (i * instruction_line_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * (w/800), (200, 200, 200), 1)
        
        cv2.putText(frame, "Phat trien boi Biitech", (w - int(180 * (w/800)), h - int(20 * (h/600))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * (w/800), (100, 100, 100), 1)
        
        # Management Buttons
        if role_lower in ['admin', 'company']:
            # 1. Connect HKB (Left) - Available for both Admin and Company
            cv2.rectangle(frame, (cX - int(380*(w/800)), h - int(84*(h/600))), (cX - int(140*(w/800)), h - int(20*(h/600))), (50, 100, 50), -1) 
            cv2.putText(frame, "KET NOI HKB", (cX - int(355*(w/800)), h - int(40*(h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * (w/800), (255, 255, 255), 2)

        if role_lower == 'admin':
            # 2. Manage Users (Center) - Admin Only
            cv2.rectangle(frame, (cX - int(120*(w/800)), h - int(84*(h/600))), (cX + int(120*(w/800)), h - int(20*(h/600))), (60, 60, 180), -1)
            cv2.putText(frame, "QUAN LY USER", (cX - int(95*(w/800)), h - int(40*(h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * (w/800), (255, 255, 255), 2)

            # 3. Manage Company (Right) - Admin Only
            cv2.rectangle(frame, (cX + int(140*(w/800)), h - int(84*(h/600))), (cX + int(380*(w/800)), h - int(20*(h/600))), (100, 50, 150), -1)
            cv2.putText(frame, "QUAN LY CONG TY", (cX + int(160*(w/800)), h - int(40*(h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * (w/800), (255, 255, 255), 2)
            
        # 4. Settings Button (Top Left) - Available to all logged-in users
        cv2.rectangle(frame, (20, 15), (150, 65), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 15), (150, 65), (255, 255, 255), 1)
        cv2.putText(frame, "CAI DAT", (45, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * (w/800), (255, 255, 255), 2)
        
        # User Info Display
        user_label = f"User: {self.session_username} ({self.session_role})"
        cv2.putText(frame, user_label, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * (w/800), (200, 200, 200), 1)

        # Logout Button (Top Right)
        logout_bg = (50, 50, 200) # Reddish-blue
        cv2.rectangle(frame, (w - 160, 15), (w - 20, 65), logout_bg, -1)
        cv2.rectangle(frame, (w - 160, 15), (w - 20, 65), (255, 255, 255), 1)
        cv2.putText(frame, "DANG XUAT", (w - 150, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * (w/800), (255, 255, 255), 2)
        
        self.frame = frame
        return frame

    @staticmethod
    def get_user_form(include_upload=False, session_role=None, mongo_db=None):
        """
        Opens a centered tkinter dialog to collect User ID, Name, Birthday, Company, and optionally Photos.
        Returns: (ID, Name, Birthday, file_paths, company_id) or None if cancelled.
        """
        import tkinter as tk
        from tkinter import messagebox, filedialog, ttk
        
        logger.info("Initializing enrollment form UI...")
        root = tk.Tk()
        root.title("Form Đăng Ký Người Dùng")
        
        # Center the window
        base_h = 300
        if include_upload: base_h += 100
        if str(session_role).lower() == 'admin': base_h += 60
        
        window_width, window_height = 380, base_h
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
        tk.Label(root, text="ĐĂNG KÝ THÔNG TIN", font=("Arial", 12, "bold")).pack(pady=10)
        
        # ID Selection / Entry
        tk.Label(root, text="Mã nhân viên *:").pack()
        entry_id = tk.Entry(root, width=30)
        entry_id.pack(pady=2)

        tk.Label(root, text="Họ và tên *:").pack()
        entry_name = tk.Entry(root, width=30)
        entry_name.pack(pady=2)

        tk.Label(root, text="Ngày sinh (DD/MM/YYYY):").pack()
        entry_bday = tk.Entry(root, width=30)
        entry_bday.pack(pady=2)

        # Company selection for admin
        combo_cid = None
        company_map = {}
        
        if str(session_role).lower() == 'admin' and mongo_db:
            tk.Label(root, text="Phân quyền Công ty *:").pack(pady=(5, 0))
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
                
                combo_cid = ttk.Combobox(root, values=company_display_list, width=27, state="readonly")
                combo_cid.set("Admin (admin)")
                combo_cid.pack(pady=2)
            except Exception as e:
                logger.error(f"UI: Error loading companies: {e}")

        # File upload section
        lbl_file_count = None
        if include_upload:
            tk.Label(root, text="Ảnh khuôn mặt *:").pack(pady=(10, 0))
            
            def on_select_files():
                logger.info("Opening file dialog for image selection...")
                files = filedialog.askopenfilenames(
                    title="Chọn ảnh khuôn mặt",
                    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
                )
                if files:
                    form_data["files"] = list(files)
                    lbl_file_count.config(text=f"Đã chọn {len(files)} ảnh", fg="green")
                    logger.info(f"User selected {len(files)} files.")
                else:
                    logger.info("File selection cancelled.")
            
            tk.Button(root, text="Chọn ảnh", command=on_select_files).pack(pady=2)
            lbl_file_count = tk.Label(root, text="Chưa chọn ảnh", fg="gray")
            lbl_file_count.pack()

        def on_submit():
            logger.info("Enrollment form submit clicked.")
            u_id = entry_id.get().strip()
            u_name = entry_name.get().strip()
            u_bday = entry_bday.get().strip()
            
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
                form_data["company_id"] = None # Will be set by caller or session
            
            form_data["id"] = u_id
            form_data["name"] = u_name
            form_data["bday"] = u_bday or "N/A"
            logger.info(f"Form data validated for {u_name}. Closing form.")
            root.destroy()

        # Submit button
        tk.Button(root, text="XÁC NHẬN", command=on_submit, width=15, bg="#28a745", fg="white").pack(pady=15)
        
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        logger.info("Starting form mainloop...")
        root.mainloop()
        
        if form_data["id"] is None:
            logger.info("Enrollment form closed without submission.")
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
        root.withdraw()
        u_id = simpledialog.askstring("Chỉnh sửa", "Nhập Mã nhân viên cần chỉnh sửa:", parent=root)
        root.destroy()
        return u_id

    @staticmethod
    def get_edit_user_form(current_id, current_name, current_bday, session_role=None):
        """
        Dialog to edit Name, Birthday, and optionally enroll face.
        """
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.title("Chỉnh sửa thông tin")
        
        window_width, window_height = 400, 420
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
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
            result["bday"] = entry_bday.get().strip()
            if not result["name"]:
                messagebox.showwarning("Cảnh báo", "Họ tên không được để trống!")
                return
            root.destroy()

        def on_delete():
            if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa nhân viên này?"):
                result["delete"] = True
                root.destroy()
        
        def on_enroll_camera():
            """Mark for camera enrollment after saving."""
            result["name"] = entry_name.get().strip()
            result["bday"] = entry_bday.get().strip()
            if not result["name"]:
                messagebox.showwarning("Cảnh báo", "Họ tên không được để trống!")
                return
            result["enroll_camera"] = True
            root.destroy()
        
        def on_enroll_upload():
            """Mark for file upload enrollment after saving."""
            result["name"] = entry_name.get().strip()
            result["bday"] = entry_bday.get().strip()
            if not result["name"]:
                messagebox.showwarning("Cảnh báo", "Họ tên không được để trống!")
                return
            result["enroll_upload"] = True
            root.destroy()

        is_company = (str(session_role).lower() == 'company')

        tk.Label(root, text=f"ID: {current_id}", font=("Arial", 10, "bold")).pack(pady=10)
        
        tk.Label(root, text="Họ và tên:").pack()
        entry_name = tk.Entry(root, width=35)
        entry_name.insert(0, current_name)
        if is_company: entry_name.config(state='readonly')
        entry_name.pack(pady=2)

        tk.Label(root, text="Ngày sinh (DD/MM/YYYY):").pack()
        entry_bday = tk.Entry(root, width=35)
        entry_bday.insert(0, current_bday)
        if is_company: entry_bday.config(state='readonly')
        entry_bday.pack(pady=2)

        # Separator
        tk.Label(root, text="─" * 50, fg="gray").pack(pady=10)
        tk.Label(root, text="ĐĂNG KÝ KHUÔN MẶT", font=("Arial", 9, "bold"), fg="#17a2b8").pack()
        
        # Face enrollment buttons
        face_btn_frame = tk.Frame(root)
        face_btn_frame.pack(pady=5)
        
        tk.Button(face_btn_frame, text="📷 Chụp bằng Camera", command=on_enroll_camera, 
                 width=18, bg="#007bff", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(face_btn_frame, text="📁 Upload từ File", command=on_enroll_upload, 
                 width=18, bg="#6c757d", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        # Separator
        tk.Label(root, text="─" * 50, fg="gray").pack(pady=10)
        
        # Action buttons
        btn_save = tk.Button(root, text="LƯU THAY ĐỔI", command=on_save, width=25, bg="#28a745", fg="white")
        btn_delete = tk.Button(root, text="XÓA NHÂN VIÊN", command=on_delete, width=25, bg="#dc3545", fg="white")
        
        if is_company:
            btn_save.config(state='disabled', bg='#6c757d')
            btn_delete.config(state='disabled', bg='#6c757d')
            tk.Label(root, text="* Quyền Company chỉ được phép cập nhật Ảnh Detect", fg="red", font=("Arial", 8)).pack()

        btn_save.pack(pady=5)
        btn_delete.pack(pady=5)
        
        root.mainloop()
        return result if result["name"] or result["delete"] or result["enroll_camera"] or result["enroll_upload"] else None

    @staticmethod
    def show_user_list_ui(user_list):
        """
        Displays a table of all enrolled users using tkinter.
        """
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Danh sách nhân viên đã đăng ký")
        
        window_width, window_height = 500, 400
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        root.attributes('-topmost', True)

        label = tk.Label(root, text=f"Tổng cộng: {len(user_list)} nhân viên", font=("Arial", 11, "bold"))
        label.pack(pady=10)

        # Create Treeview
        columns = ("id", "name", "birthday", "face_status")
        tree = ttk.Treeview(root, columns=columns, show="headings")
        
        tree.heading("id", text="Mã nhân viên")
        tree.heading("name", text="Họ và tên")
        tree.heading("birthday", text="Ngày sinh")
        tree.heading("face_status", text="Khuôn mặt")
        
        tree.column("id", width=120)
        tree.column("name", width=200)
        tree.column("birthday", width=120)
        tree.column("face_status", width=100, anchor="center")

        for user in user_list:
            # Handle both old format (dict with user_id, user_name, birthday) 
            # and new format (dict with user_id, user_name, birthday, has_face)
            u_id = user.get("user_id", "N/A")
            u_name = user.get("user_name") or user.get("name", "Unknown")
            u_bday = user.get("birthday", "N/A")
            has_face = user.get("has_face", True)  # Default True for backward compatibility
            face_status = "✓ Đã đăng ký" if has_face else "⚠ Chưa có"
            
            tree.insert("", tk.END, values=(u_id, u_name, u_bday, face_status))

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        btn_close = tk.Button(root, text="ĐÓNG", command=root.destroy, width=15, bg="#007bff", fg="white")
        btn_close.pack(pady=10)

        root.mainloop()

    @staticmethod
    def pick_company_ui(companies):
        """
        Dialog to select a company from a list.
        Returns company_id or None.
        """
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
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

        # Add "ALL" option for convenience
        tree.insert("", tk.END, values=("ALL", "--- TẤT CẢ CÔNG TY ---"))
        
        for c in companies:
            tree.insert("", tk.END, values=(c.get('company_id'), c.get('name')))

        def on_select():
            sel = tree.selection()
            if sel:
                selected["id"] = tree.item(sel[0])['values'][0]
                root.destroy()

        tree.pack(padx=10, pady=10, fill="both", expand=True)
        tk.Button(root, text="XÁC NHẬN", command=on_select, bg="#28a745", fg="white", width=15).pack(pady=10)

        root.mainloop()
        return selected["id"]

    @staticmethod
    def pick_user_ui(user_list):
        """
        Dialog to select a user from a list.
        Returns user_id or None.
        """
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
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

        root.mainloop()
        return selected["id"]

    @staticmethod
    def get_date_form(title="Chọn ngày"):
        """
        Dialog to select a date. Returns string YYYY-MM-DD or None.
        """
        import tkinter as tk
        from datetime import datetime
        
        root = tk.Tk()
        root.title(title)
        root.geometry("300x180")
        root.attributes('-topmost', True)
        
        result = {"date": None}
        today = datetime.now().strftime("%Y-%m-%d")
        
        tk.Label(root, text="NHẬP NGÀY CẦN XEM", font=("Arial", 11, "bold")).pack(pady=15)
        
        tk.Label(root, text="Định dạng: YYYY-MM-DD").pack()
        entry_date = tk.Entry(root, width=20)
        entry_date.insert(0, today)
        entry_date.pack(pady=5)
        
        def on_ok():
            date_str = entry_date.get().strip()
            # Simple validation
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                result["date"] = date_str
                root.destroy()
            except ValueError:
                from tkinter import messagebox
                messagebox.showerror("Lỗi", "Định dạng ngày không hợp lệ! Vui lòng nhập YYYY-MM-DD")

        def on_cancel():
            root.destroy()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="XÁC NHẬN", command=on_ok, bg="green", fg="white", width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="HỦY", command=on_cancel, width=12).pack(side=tk.LEFT, padx=5)
        
        root.mainloop()
        return result["date"]

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
            root.transient(parent)
            root.grab_set()
        else:
            root = tk.Tk()
            
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

    @staticmethod
    def show_attendance_logs_ui(logs, title="Lịch sử điểm danh", session_role=None, session_user_id=None, session_username="GLOBAL"):
        """
        Displays a table of attendance logs using tkinter.
        """
        import tkinter as tk
        from tkinter import ttk, messagebox

        root = tk.Tk()
        root.title(title)
        
        window_width, window_height = 800, 550
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        root.attributes('-topmost', True)

        label = tk.Label(root, text=f"{title} ({len(logs)} lượt)", font=("Arial", 11, "bold"))
        label.pack(pady=10)

        # Create Treeview with checkboxes via selectmode
        columns = ("id", "user_id", "name", "time", "status", "uploaded")
        tree = ttk.Treeview(root, columns=columns, show="headings", selectmode="extended")
        
        tree.heading("id", text="ID")
        tree.heading("user_id", text="Mã NV")
        tree.heading("name", text="Họ và tên")
        tree.heading("time", text="Thời gian")
        tree.heading("status", text="Trạng thái")
        tree.heading("uploaded", text="Đã upload")
        
        tree.column("id", width=50)
        tree.column("user_id", width=100)
        tree.column("name", width=150)
        tree.column("time", width=150)
        tree.column("status", width=80)
        tree.column("uploaded", width=100)

        # logs structure from MongoDB: (id, user_id, user_name, timestamp, date, status, image_path, uploaded_to)
        for log in logs:
            status_val = log[5] if len(log) > 5 else "N/A"
            uploaded_to = log[7] if len(log) > 7 else []
            uploaded_str = "✓" if uploaded_to else ""
            tree.insert("", tk.END, values=(log[0], log[1], log[2], log[3], status_val, uploaded_str))

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        def on_batch_upload():
            """Upload selected or all attendance logs to a chosen system."""
            from src.services.hkb_service import hkb_service
            from src.attendance.mongodb_mgr import mongo_db
            
            logger.info("Batch upload button clicked.")
            
            # 1. COLLECT ALL DATA FROM TREE IMMEDIATELY before opening any other windows
            selected_items = tree.selection()
            if not selected_items:
                if not messagebox.askyesno("Xác nhận", "Không có dòng nào được chọn. Upload tất cả?"):
                    return
                selected_items = tree.get_children()
            
            if not selected_items:
                messagebox.showwarning("!", "Không có dữ liệu để upload")
                return
            
            # Map item data while the tree is still valid
            log_items_batch = []
            for item in selected_items:
                log_items_batch.append({
                    "id_from_tree": str(tree.item(item)['values'][0]),
                    "item_id": item
                })

            # 2. Get list of connected systems for this user (user-specific keys)
            user_id = session_user_id or 1
            connections = hkb_service.get_connections(user_id=user_id, username=session_username)
            
            if not connections or not isinstance(connections, list):
                messagebox.showerror("Lỗi", "Không tìm thấy hệ thống đã kết nối")
                return
            
            # Filter only connected systems
            connected_systems = [c for c in connections if c.get("client_register") == 1]
            if not connected_systems:
                messagebox.showwarning("!", "Chưa có hệ thống nào được kết nối")
                return
            
            # 3. Show system selection dialog (passing main root for Toplevel use)
            selected_system = AttendanceUI.pick_system_for_upload(connected_systems, parent=root)
            if not selected_system:
                return
            
            # 4. Prepare logs data for upload using the pre-collected IDs
            upload_logs = []
            log_ids = []
            
            for item_info in log_items_batch:
                log_id = item_info["id_from_tree"]
                log_data = next((l for l in logs if str(l[0]) == log_id), None)
                
                if log_data:
                    log_ids.append(log_id)
                    upload_logs.append({
                        "session_id": log_data[6] if len(log_data) > 6 else log_id,
                        "user_id": log_data[1],
                        "user_name": log_data[2],
                        "timestamp": log_data[3],
                        "status": log_data[5] if len(log_data) > 5 else "IN",
                        "image_webp": mongo_db.get_log_image(log_id)
                    })
            
            if not upload_logs:
                messagebox.showwarning("!", "Không có dữ liệu hợp lệ để upload")
                return
            
            # Get auth info for selected system
            auth_data = mongo_db.auth_services.find_one({"uuid": selected_system["system_id"]})
            if not auth_data:
                messagebox.showerror("Lỗi", "Không tìm thấy thông tin xác thực cho hệ thống này")
                return
            
            # Upload
            root.config(cursor="watch")
            root.update()
            
            result = hkb_service.upload_timekeepers(
                endpoint=selected_system["endpoint"],
                system_id=selected_system["system_id"],
                api_key=auth_data["key"],
                user_id=auth_data.get("user_id", user_id),
                attendance_logs=upload_logs
            )
            
            root.config(cursor="")
            
            if result and result.success:
                # Mark as uploaded
                mongo_db.mark_logs_uploaded(log_ids, selected_system["system_id"])
                messagebox.showinfo("Thành công", f"Đã upload {len(upload_logs)} lượt chấm công lên {selected_system['name']}")
                root.destroy()
            else:
                err_msg = result.message if result and result.message else "Upload thất bại"
                messagebox.showerror("Lỗi", err_msg)

        def on_view_image():
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn một lượt điểm danh để xem ảnh!")
                return
            
            from src.attendance.mongodb_mgr import mongo_db
            import cv2
            import numpy as np

            item_values = tree.item(selected_item[0])['values']
            log_id = str(item_values[0])
            user_name = item_values[2]
            
            # Fetch image from MongoDB
            img_bytes = mongo_db.get_log_image(log_id)
            if img_bytes:
                # Decode WebP/Image from bytes
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Show image in a new OpenCV window
                    win_title = f"Anh diem danh: {user_name} ({log_id})"
                    cv2.imshow(win_title, img)
                    cv2.waitKey(1) # Keep window responsive
                else:
                    messagebox.showerror("Lỗi", "Không thể hiển thị dữ liệu ảnh.")
            else:
                messagebox.showwarning("Thông báo", "Lượt điểm danh này không có dữ liệu ảnh hoặc ảnh đã bị xóa.")

        btn_container = tk.Frame(root)
        btn_container.pack(pady=10)

        tk.Button(btn_container, text="XEM ẢNH", command=on_view_image, width=15, bg="#f39c12", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Show batch upload button for admin and company roles
        if session_role and str(session_role).lower() in ['admin', 'company']:
            tk.Button(btn_container, text="UPLOAD LÊN HỆ THỐNG", command=on_batch_upload, width=20, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_container, text="ĐÓNG", command=root.destroy, width=15, bg="#007bff", fg="white").pack(side=tk.LEFT, padx=5)

        root.mainloop()

    def show_hkb_connections_ui(self):
        """
        Displays a list of HKB connections and allows registering/connecting.
        """
        import tkinter as tk
        from tkinter import ttk, messagebox
        from src.services.hkb_service import hkb_service
        from src.config import AuthServiceConfig

        # Lấy user_id từ session đang đăng nhập, mặc định là 1
        current_user_id = getattr(self, "session_user_id", 1) or 1

        root = tk.Tk()
        root.title("Kết nối AuthService")
        root.geometry("1000x500")
        root.attributes('-topmost', True)

        tk.Label(root, text="DANH SÁCH KẾT NỐI", font=("Arial", 14, "bold")).pack(pady=10)

        # Create Treeview with more columns
        columns = ("id", "name", "system_id", "endpoint", "actived", "status")
        tree = ttk.Treeview(root, columns=columns, show="headings")
        
        tree.heading("id", text="ID")
        tree.heading("name", text="Tên hệ thống")
        tree.heading("system_id", text="System ID")
        tree.heading("endpoint", text="Endpoint")
        tree.heading("actived", text="Hoạt động")
        tree.heading("status", text="Trạng thái")
        
        tree.column("id", width=40, anchor="center")
        tree.column("name", width=250)
        tree.column("system_id", width=200)
        tree.column("endpoint", width=200)
        tree.column("actived", width=80, anchor="center")
        tree.column("status", width=120, anchor="center")

        def refresh_list():
            for item in tree.get_children():
                tree.delete(item)
            
            # Truyền user_id hiện tại để lọc danh sách UUID đã đăng ký, và session_username cho group keys
            connections = hkb_service.get_connections(
                user_id=current_user_id, 
                username=getattr(self, "session_username", "GLOBAL")
            )
            if connections and isinstance(connections, list):
                for conn in connections:
                    # Logic: client_register = 0 -> Chưa kết nối, 1 -> Đã kết nối
                    is_registered = conn.get("client_register", 0)
                    status_str = "Đã kết nối" if is_registered == 1 else "Chưa kết nối"
                    
                    tree.insert("", tk.END, values=(
                        conn.get("id", "N/A"),
                        conn.get("name", "N/A"),
                        conn.get("system_id", "N/A"),
                        conn.get("endpoint", "N/A"),
                        "Có" if conn.get("actived") == 1 else "Không",
                        status_str
                    ))
            elif connections:
                # Might be a single dict or other structure
                logger.info(f"API result is not a list: {connections}")
            else:
                messagebox.showinfo("Thông báo", "Không tìm thấy kết nối nào hoặc lỗi API.")

        def on_register():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("!", "Vui lòng chọn hệ thống cần kết nối từ danh sách")
                return
            
            # Get data from selected row
            item = tree.item(sel[0])['values']
            conn_id = item[0]
            conn_name = item[1]
            remote_sys_id = item[2]

            # Gửi yêu cầu kết nối với thông tin hệ thống được chọn
            root.config(cursor="watch") # Đổi chuột sang trạng thái chờ
            root.update()
            
            result = hkb_service.register_client(
                system_id=remote_sys_id,
                external_id=current_user_id,
                description=f"đang yêu cầu kết nối hệ thống {conn_name}",
                user_info={"app": "Face Attendance System"},
                system_connection_id=conn_id,
                system_register=AuthServiceConfig.SYSTEM_ID
            )
            
            root.config(cursor="") # Trả lại chuột bình thường
            
            if result and result.success:
                msg = result.message if result.message else f"Đã gửi yêu cầu kết nối tới: {conn_name}"
                messagebox.showinfo("Thành công", msg)
                refresh_list()
            else:
                err_msg = result.message if result and result.message else "Gửi yêu cầu kết nối thất bại"
                messagebox.showerror("Lỗi", err_msg)

        def on_revoke():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("!", "Vui lòng chọn hệ thống cần hủy kết nối")
                return
            
            item = tree.item(sel[0])['values']
            conn_name = item[1]
            remote_sys_id = item[2]
            status_str = item[5]

            if status_str != "Đã kết nối":
                messagebox.showwarning("!", "Hệ thống này chưa được kết nối!")
                return

            if not messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn hủy kết nối với {conn_name}?\nHành động này sẽ vô hiệu hóa API Key hiện tại."):
                return

            # Lấy API Key từ MongoDB cục bộ
            auth_data = mongo_db.auth_services.find_one({"uuid": remote_sys_id})
            if not auth_data or not auth_data.get("key"):
                messagebox.showerror("Lỗi", "Không tìm thấy API Key cục bộ để thực hiện hủy!")
                return
            
            api_key = auth_data["key"]

            # Mặc định mật khẩu là 123 khi hủy kết nối
            password = "123"

            root.config(cursor="watch")
            root.update()
            
            result = hkb_service.revoke_connection(
                system_id=remote_sys_id,
                api_key=api_key,
                password=password
            )
            
            root.config(cursor="")
            
            if result and result.success:
                messagebox.showinfo("Thành công", f"Đã hủy kết nối thành công với: {conn_name}")
                refresh_list()
            else:
                err_msg = result.message if result and result.message else "Hủy kết nối thất bại"
                messagebox.showerror("Lỗi", err_msg)

        def on_get_employees():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("!", "Vui lòng chọn hệ thống để lấy danh sách nhân sự")
                return
            
            item = tree.item(sel[0])['values']
            conn_name = item[1]
            remote_sys_id = item[2]
            endpoint = item[3]
            status_str = item[5]

            if status_str != "Đã kết nối":
                messagebox.showwarning("!", "Hệ thống này chưa được kết nối! Vui lòng đăng ký trước.")
                return

            # Lấy API Key từ MongoDB cục bộ
            auth_data = mongo_db.auth_services.find_one({"uuid": remote_sys_id})
            if not auth_data or not auth_data.get("key"):
                messagebox.showerror("Lỗi", "Không tìm thấy API Key cục bộ để thực hiện lấy dữ liệu!")
                return
            
            api_key = auth_data["key"]
            ext_id = auth_data.get("user_id", current_user_id)

            root.config(cursor="watch")
            root.update()
            
            result = hkb_service.get_employees(
                endpoint=endpoint,
                system_id=remote_sys_id,
                api_key=api_key,
                user_id=ext_id
            )
            
            root.config(cursor="")
            
            if result and result.success:
                employees = result.data
                if not employees or not isinstance(employees, list):
                    messagebox.showinfo("Thông báo", "Không có dữ liệu nhân sự hoặc định dạng không đúng.")
                    return
                
                # Determine target company ID for saving
                target_cid = remote_sys_id
                if str(self.session_role).lower() == 'company' and self.session_company_id:
                    target_cid = self.session_company_id
                    logger.info(f"Company user sync: using forced company_id '{target_cid}' instead of system uuid '{remote_sys_id}'")
                
                # Hiển thị danh sách nhân sự trong một cửa sổ mới
                self.show_remote_employees_ui(conn_name, employees, target_cid)
            else:
                err_msg = result.message if result and result.message else "Lấy danh sách nhân sự thất bại"
                if result and result.data and "raw" in result.data:
                    logger.debug(f"Raw response: {result.data['raw']}")
                messagebox.showerror("Lỗi", err_msg)

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        btn_container = tk.Frame(root)
        btn_container.pack(pady=10)

        tk.Button(btn_container, text="LÀM MỚI", command=refresh_list, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐĂNG KÝ HỆ THỐNG", command=on_register, width=15, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="LẤY DANH SÁCH NV", command=on_get_employees, width=15, bg="#17a2b8", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="HỦY KẾT NỐI", command=on_revoke, width=15, bg="#dc3545", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐÓNG", command=root.destroy, width=15, bg="#007bff", fg="white").pack(side=tk.LEFT, padx=5)

        refresh_list()
        root.mainloop()

    def show_remote_employees_ui(self, system_name, employees, target_company_id):
        """
        Displays a list of employees fetched from a remote system with saving options.
        """
        import tkinter as tk
        from tkinter import ttk, messagebox

        root = tk.Tk()
        root.title(f"Nhân viên từ {system_name}")
        root.geometry("900x550")
        root.attributes('-topmost', True)

        tk.Label(root, text=f"DANH SÁCH NHÂN VIÊN - {system_name}", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(root, text=f"Công ty: {target_company_id}", font=("Arial", 10, "italic"), fg="gray").pack()
        tk.Label(root, text=f"Tìm thấy: {len(employees)} nhân sự", font=("Arial", 10)).pack(pady=5)

        # Create Treeview with multiple selection enabled (default)
        columns = ("user_id", "name", "bday", "sex", "group")
        tree = ttk.Treeview(root, columns=columns, show="headings", selectmode="extended")
        
        tree.heading("user_id", text="Mã nhân viên")
        tree.heading("name", text="Họ và tên")
        tree.heading("bday", text="Ngày sinh")
        tree.heading("sex", text="Giới tính")
        tree.heading("group", text="Nhóm/Phòng ban")
        
        tree.column("user_id", width=100, anchor="center")
        tree.column("name", width=200)
        tree.column("bday", width=100, anchor="center")
        tree.column("sex", width=80, anchor="center")
        tree.column("group", width=150)

        # Store full employee data for lookup
        emp_map = {}
        for emp in employees:
            u_id = emp.get("barcode") or emp.get("user_id") or emp.get("uid") or emp.get("id") or "N/A"
            u_name = emp.get("full_name") or emp.get("name") or emp.get("user_name") or "Unknown"
            u_bday = emp.get("birthday") or emp.get("birth_day") or "N/A"
            u_sex = emp.get("sex") or "-"
            u_group = emp.get("group_id") or emp.get("company_id") or emp.get("description") or "-"
            
            item_id = tree.insert("", tk.END, values=(u_id, u_name, u_bday, u_sex, u_group))
            emp_map[item_id] = {
                "id": u_id,
                "name": u_name,
                "bday": u_bday,
                "cid": target_company_id
            }

        tree.pack(expand=True, fill="both", padx=10, pady=10)

        def save_to_db(selected_only=False):
            if selected_only:
                items = tree.selection()
                if not items:
                    messagebox.showwarning("!", "Vui lòng chọn ít nhất một nhân viên")
                    return
                targets = [emp_map[i] for i in items]
                msg_confirm = f"Lưu {len(targets)} nhân viên đã chọn vào hệ thống?"
            else:
                targets = list(emp_map.values())
                msg_confirm = f"Lưu TẤT CẢ {len(targets)} nhân viên vào hệ thống?"

            if not messagebox.askyesno("Xác nhận", msg_confirm):
                return
            
            saved_count = 0
            errors = []
            update_all = False
            skip_all = False
            
            for t in targets:
                # Try saving normally
                ok, msg = mongo_db.save_employee(t["id"], t["name"], t["bday"], t["cid"], force_update=False)
                
                if not ok and "đã tồn tại" in msg:
                    # User already exists
                    if skip_all:
                        continue
                    if update_all:
                        ok, msg = mongo_db.save_employee(t["id"], t["name"], t["bday"], t["cid"], force_update=True)
                    else:
                        # Ask the user what to do
                        # Since we might have many, we offer "Update All" or "Skip All" via a custom or multiple choice
                        # For simplicity with basic messagebox, we'll ask Yes/No/Cancel
                        # Yes -> Update this one, No -> Skip this one, Cancel -> Stop
                        
                        confirm_msg = f"Nhân viên ID '{t['id']}' ({t['name']}) đã tồn tại.\n\nBạn có muốn CẬP NHẬT thông tin mới nhất không?"
                        
                        # Use a more flexible dialog if possible, or just ask yes/no
                        # To support "Update All", we can use askyesnocancel or a custom dialog.
                        # Let's try a simple approach with a count-save pop-up
                        
                        res = messagebox.askyesnocancel("Phát hiện trùng lặp", confirm_msg)
                        
                        if res is True: # Yes: Update
                            ok, msg = mongo_db.save_employee(t["id"], t["name"], t["bday"], t["cid"], force_update=True)
                        elif res is False: # No: Skip
                            continue
                        else: # None: Cancel Batch
                            logger.info("Batch save cancelled by user.")
                            break
                
                if ok:
                    saved_count += 1
                    # Also sync to Qdrant if user exists there
                    try:
                        from src.attendance.qdrant_db import attendance
                        user_info = attendance.get_user_info(t["id"])
                        if user_info:
                            # User exists in Qdrant, update their info
                            attendance.update_user_info(t["id"], t["name"], t["bday"])
                            logger.info(f"Synced info to Qdrant for user {t['id']}")
                    except Exception as e:
                        logger.warning(f"Could not sync to Qdrant for {t['id']}: {e}")
                else:
                    errors.append(f"ID {t['id']}: {msg}")
            
            if errors:
                error_msg = "\n".join(errors[:10])
                if len(errors) > 10: error_msg += f"\n... và {len(errors)-10} lỗi khác"
                messagebox.showwarning("Kết quả lưu", f"Đã lưu/cập nhật {saved_count}/{len(targets)} nhân viên.\n\nCác lỗi:\n{error_msg}")
            else:
                messagebox.showinfo("Thành công", f"Đã lưu/cập nhật thành công {saved_count}/{len(targets)} nhân viên vào công ty {target_company_id}")
            
            if saved_count > 0:
                root.destroy()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="LƯU ĐÃ CHỌN", command=lambda: save_to_db(True), width=20, bg="#28a745", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="LƯU TẤT CẢ", command=lambda: save_to_db(False), width=20, bg="#17a2b8", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="HỦY", command=root.destroy, width=15).pack(side=tk.LEFT, padx=10)

        root.mainloop()

    @staticmethod
    def show_company_management_ui(mongo_db):
        """Management UI for Companies."""
        import tkinter as tk
        from tkinter import ttk, messagebox, simpledialog

        root = tk.Tk()
        root.title("Quản lý Công ty")
        root.geometry("600x500")
        root.attributes('-topmost', True)

        tk.Label(root, text="DANH SÁCH CÔNG TY", font=("Arial", 12, "bold")).pack(pady=10)

        tree = ttk.Treeview(root, columns=("id", "name", "desc"), show="headings")
        tree.heading("id", text="Company ID")
        tree.heading("name", text="Tên công ty")
        tree.heading("desc", text="Mô tả")
        
        tree.column("id", width=120)
        tree.column("name", width=200)
        tree.column("desc", width=200)

        def refresh():
            for i in tree.get_children(): tree.delete(i)
            for c in mongo_db.get_all_companies():
                tree.insert("", tk.END, values=(c.get('company_id'), c.get('name'), c.get('description')))

        def on_add():
            add_win = tk.Toplevel(root)
            add_win.title("Thêm Công ty")
            add_win.geometry("300x250")
            
            tk.Label(add_win, text="Company ID:").pack()
            e_id = tk.Entry(add_win); e_id.pack()
            tk.Label(add_win, text="Tên công ty:").pack()
            e_name = tk.Entry(add_win); e_name.pack()
            tk.Label(add_win, text="Mô tả:").pack()
            e_desc = tk.Entry(add_win); e_desc.pack()
            
            def submit():
                cid, name, desc = e_id.get(), e_name.get(), e_desc.get()
                if not cid or not name: return messagebox.showwarning("!", "Nhập ID & Tên")
                success, msg = mongo_db.create_company(cid, name, desc)
                if success:
                    messagebox.showinfo("OK", "Đã thêm công ty")
                    add_win.destroy()
                    refresh()
                else: messagebox.showerror("Lỗi", msg)
            
            tk.Button(add_win, text="LƯU", command=submit, bg="green", fg="white").pack(pady=10)

        def on_delete():
            sel = tree.selection()
            if not sel: return
            cid = tree.item(sel[0])['values'][0]
            if messagebox.askyesno("Xác nhận", f"Xóa công ty {cid}?"):
                if mongo_db.delete_company(cid):
                    messagebox.showinfo("OK", "Đã xóa")
                    refresh()
                else: messagebox.showerror("Lỗi", "Không thể xóa")

        tree.pack(fill="both", expand=True, padx=10)
        btn_frame = tk.Frame(root); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="LÀM MỚI", command=refresh).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="THÊM MỚI", command=on_add, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="XÓA", command=on_delete, bg="red", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="ĐÓNG", command=root.destroy).pack(side=tk.LEFT, padx=5)

        refresh()
        root.mainloop()

    @staticmethod
    def show_user_management_ui(mongo_db):
        """Management UI for Cloud Users (Logins)."""
        import tkinter as tk
        from tkinter import ttk, messagebox

        root = tk.Tk()
        root.title("Quản lý User Hệ thống")
        root.geometry("600x500")
        root.attributes('-topmost', True)

        tk.Label(root, text="DANH SÁCH USER (ADMIN/COMPANY)", font=("Arial", 12, "bold")).pack(pady=10)

        tree = ttk.Treeview(root, columns=("user", "role", "company"), show="headings")
        tree.heading("user", text="Username")
        tree.heading("role", text="Quyền")
        tree.heading("company", text="Phân quyền Công ty")
        
        # Helper for display names in the list
        all_companies = mongo_db.get_all_companies()
        company_id_to_name = {c.get("company_id"): c.get("name", c.get("company_id")) for c in all_companies}
        company_id_to_name["admin"] = "Admin (Cổng Tổng)"

        def refresh():
            for i in tree.get_children(): tree.delete(i)
            for u in mongo_db.get_all_cloud_users():
                cid = u.get('company_id', 'admin')
                cname = company_id_to_name.get(cid, cid)
                tree.insert("", tk.END, values=(u.get('username'), u.get('role'), cname))

        def on_add():
            from tkinter import ttk
            add_win = tk.Toplevel(root); add_win.title("Thêm User Mới"); add_win.geometry("300x420")
            
            tk.Label(add_win, text="Username:").pack(pady=5)
            e_user = tk.Entry(add_win); e_user.pack()
            
            tk.Label(add_win, text="Password:").pack(pady=5)
            e_pwd = tk.Entry(add_win, show="*"); e_pwd.pack()
            
            tk.Label(add_win, text="Quyền hạn:").pack(pady=5)
            e_role = ttk.Combobox(add_win, values=["admin", "company"])
            e_role.set("company"); e_role.pack()
            
            tk.Label(add_win, text="Phân quyền Công ty:").pack(pady=5)
            
            # Map display names to IDs
            company_map = {"Admin (Cổng Tổng)": "admin"}
            company_display_list = ["Admin (Cổng Tổng)"]
            
            for c in all_companies:
                cid = c.get("company_id")
                cname = c.get("name", cid)
                display_text = f"{cname} ({cid})"
                company_map[display_text] = cid
                company_display_list.append(display_text)
            
            e_cid = ttk.Combobox(add_win, values=company_display_list, state="readonly")
            e_cid.set("Admin (Cổng Tổng)"); e_cid.pack()
            
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

            tk.Button(add_win, text="LƯU TÀI KHOẢN", command=submit, bg="#28a745", fg="white", font=("Arial", 10, "bold")).pack(pady=20)

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

        tree.pack(fill="both", expand=True, padx=10)
        btn_frame = tk.Frame(root); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="LÀM MỚI", command=refresh).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="THÊM MỚI", command=on_add, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="XÓA USER", command=on_delete, bg="red", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="ĐÓNG", command=root.destroy).pack(side=tk.LEFT, padx=5)
        
        refresh()
        root.mainloop()

    def show_login_dialog(self):
        """
        Shows a login dialog for admin authentication.
        """
        import tkinter as tk
        from tkinter import messagebox
        from src.attendance.mongodb_mgr import mongo_db

        login_root = tk.Tk()
        login_root.title("System Login")
        window_width, window_height = 300, 200
        screen_width = login_root.winfo_screenwidth()
        screen_height = login_root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        login_root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        login_root.attributes('-topmost', True)
        
        login_status = {"authenticated": False}

        def attempt_login():
            user = entry_user.get()
            pwd = entry_pwd.get()
            
            # Authenticate using Cloud MongoDB
            auth_info = mongo_db.verify_login(user, pwd)
            
            if auth_info:
                login_status["authenticated"] = True
                self.is_admin_logged_in = True # Keeping the flag name but role based now
                self.session_role = auth_info["role"]
                self.session_company_id = auth_info["company_id"]
                self.session_username = auth_info["username"]
                self.session_user_id = auth_info.get("user_id", 1) # Lấy user_id từ DB nếu có
                login_root.destroy()
            else:
                messagebox.showerror("Lỗi", "Sai tài khoản hoặc mật khẩu Cloud!")

        tk.Label(login_root, text="ĐĂNG NHẬP HỆ THỐNG", font=("Arial", 10, "bold")).pack(pady=10)
        tk.Label(login_root, text="Tên đăng nhập:").pack()
        entry_user = tk.Entry(login_root)
        entry_user.pack()
        tk.Label(login_root, text="Mật khẩu:").pack()
        entry_pwd = tk.Entry(login_root, show="*")
        entry_pwd.pack()
        
        tk.Button(login_root, text="ĐĂNG NHẬP", command=attempt_login, bg="#007bff", fg="white").pack(pady=10)

        login_root.mainloop()
        return login_status["authenticated"]

    @staticmethod
    def show_system_settings_ui(mongo_db, session_username="GLOBAL"):
        """UI to manage settings like Group Keys and Camera (user-specific)."""
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.title("Cài đặt Hệ thống")
        root.geometry("600x550")
        root.attributes('-topmost', True)

        tk.Label(root, text="CẤU HÌNH HỆ THỐNG", font=("Arial", 14, "bold")).pack(pady=20)

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=20)

        # --- A. CAMERA CONFIG (New) ---
        tk.Label(main_frame, text="CẤU HÌNH CAMERA (RTSP)", font=("Arial", 10, "bold"), fg="blue").pack(anchor="w", pady=(0, 5))
        
        cam_frame = tk.Frame(main_frame)
        cam_frame.pack(fill="x", pady=5)
        
        # Grid for camera fields
        tk.Label(cam_frame, text="IP Camera:").grid(row=0, column=0, sticky="e", pady=2)
        e_ip = tk.Entry(cam_frame, width=25)
        e_ip.grid(row=0, column=1, padx=5); e_ip.insert(0, mongo_db.get_setting("camera_ip", "192.168.1.1", username=session_username))
        
        tk.Label(cam_frame, text="Port (RTSP):").grid(row=0, column=2, sticky="e", pady=2)
        e_port = tk.Entry(cam_frame, width=10)
        e_port.grid(row=0, column=3, padx=5); e_port.insert(0, mongo_db.get_setting("camera_port", "554", username=session_username))
        
        tk.Label(cam_frame, text="Username:").grid(row=1, column=0, sticky="e", pady=2)
        e_user = tk.Entry(cam_frame, width=25)
        e_user.grid(row=1, column=1, padx=5); e_user.insert(0, mongo_db.get_setting("camera_user", "admin", username=session_username))
        
        tk.Label(cam_frame, text="Password:").grid(row=1, column=2, sticky="e", pady=2)
        e_pass = tk.Entry(cam_frame, width=25, show="*")
        e_pass.grid(row=1, column=3, padx=5); e_pass.insert(0, mongo_db.get_setting("camera_pass", "password", username=session_username))

        tk.Label(main_frame, text="----------------------------------------------------------", fg="gray").pack(pady=10)

        # --- B. GROUP KEYS ---
        tk.Label(main_frame, text="Group Keys (Các key cách nhau bởi dấu phẩy):", font=("Arial", 10, "bold"), fg="blue").pack(anchor="w", pady=(0, 5))
        current_keys = mongo_db.get_setting("group_keys", "", username=session_username)
        text_keys = tk.Text(main_frame, height=4, width=65)
        text_keys.pack(pady=5)
        text_keys.insert("1.0", current_keys)

        def save_settings():
            # Get values
            new_keys = text_keys.get("1.0", "end-1c").strip()
            ip = e_ip.get().strip()
            port = e_port.get().strip()
            user = e_user.get().strip()
            pwd = e_pass.get().strip()
            
            # Save all to DB
            success = True
            success &= mongo_db.set_setting("group_keys", new_keys, username=session_username)
            success &= mongo_db.set_setting("camera_ip", ip, username=session_username)
            success &= mongo_db.set_setting("camera_port", port, username=session_username)
            success &= mongo_db.set_setting("camera_user", user, username=session_username)
            success &= mongo_db.set_setting("camera_pass", pwd, username=session_username)
            
            if success:
                messagebox.showinfo("Thành công", "Đã lưu cài đặt hệ thống!\nBạn cần khởi động lại dịch vụ Camera để áp dụng thay đổi IP/Pass.")
                root.destroy()
            else:
                messagebox.showerror("Lỗi", "Không thể lưu cài đặt!")

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20, side=tk.BOTTOM)
        
        tk.Button(btn_frame, text="LƯU CÀI ĐẶT", command=save_settings, bg="#28a745", fg="white", width=20, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="HỦY", command=root.destroy, width=15).pack(side=tk.LEFT, padx=10)

        root.mainloop()

    @staticmethod
    def draw_status_bar(frame, fps, processing_time):
        """Draw FPS and AI stats."""
        cv2.putText(frame, f"FPS: {fps} | AI: {processing_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Nhan 'M' de ve Menu", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return frame
