import cv2
import numpy as np
from loguru import logger
from src.attendance.mongodb_mgr import mongo_db

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

            # 1. BAT DAU detection (Available to all)
            if col1_L < x < col1_R and cY-btn_h-gap_y < y < cY-gap_y: 
                self.current_state = STATE_DETECT
                
            # 2. DANH SACH (Available to all)
            elif col2_L < x < col2_R and cY+gap_y < y < cY+gap_y+btn_h: 
                self.current_state = STATE_LIST

            # 3. CHINH SUA (Available to all)
            elif col2_L < x < col2_R and cY-btn_h-gap_y < y < cY-gap_y: 
                self.current_state = STATE_EDIT

            # --- ADMIN & COMPANY RECOGNITION ACTIONS ---
            if str(self.session_role).lower() in ['admin', 'company']:
                # DANG KY CAM (Col 1, Row 2)
                if col1_L < x < col1_R and cY+gap_y < y < cY+gap_y+btn_h: 
                    self.current_state = STATE_ENROLL_CAM
                # DANG KY FILE (Col 1, Row 3)
                elif col1_L < x < col1_R and cY+gap_y*2+btn_h < y < cY+gap_y*2+btn_h*2: 
                    self.current_state = STATE_ENROLL_UPLOAD
                # LICH SU (Col 2, Row 3)
                elif col2_L < x < col2_R and cY+gap_y*2+btn_h < y < cY+gap_y*2+btn_h*2: 
                    self.current_state = STATE_HISTORY

            # --- ADMIN ONLY SYSTEM MANAGEMENT ---
            if str(self.session_role).lower() == 'admin':
                # --- ADMIN ONLY BOTTOM BAR ---
                if h - int(84*(h/600)) < y < h - int(20*(h/600)):
                    # 1. KET NOI HKB (Bottom Left)
                    if cX - int(380*(w/800)) < x < cX - int(140*(w/800)):
                        self.current_state = STATE_HKB_LIST
                    # 2. QUAN LY USER (Bottom Center)
                    elif cX - int(120*(w/800)) < x < cX + int(120*(w/800)):
                        self.current_state = STATE_CLOUD_USER
                    # 3. QUAN LY CONG TY (Bottom Right)
                    elif cX + int(140*(w/800)) < x < cX + int(380*(w/800)): 
                        self.current_state = STATE_COMPANY
                
                # 4. CAI DAT (Icon/Small button next to Logout or elsewhere) - TOP LEFT
                if 20 < x < 150 and 15 < y < 65:
                    self.current_state = STATE_SETTINGS

    def draw_main_menu(self, w=1280, h=720):
        """Draw a professional menu responsive to window size."""
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
        
        col1_x = cX - btn_w - gap_x
        col2_x = cX + gap_x

        # --- Column 1 ---
        # Button 1: Start System (All)
        cv2.rectangle(frame, (col1_x, cY - btn_h - gap_y), (col1_x + btn_w, cY - gap_y), (40, 180, 40), -1)
        cv2.putText(frame, "BAT DAU", (col1_x + int(90 * (w/800)), cY - gap_y - int(18 * (h/600))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)
        
        # Enrollment Buttons (Available to Admin and Company Managers)
        if self.session_role in ['admin', 'company']:
            # Button 2: Enroll Camera
            cv2.rectangle(frame, (col1_x, cY + gap_y), (col1_x + btn_w, cY + gap_y + btn_h), (200, 120, 0), -1)
            cv2.putText(frame, "DANG KY (CAM)", (col1_x + int(50 * (w/800)), cY + gap_y + int(42 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)
            
            # Button 3: Enroll Upload
            cv2.rectangle(frame, (col1_x, cY + gap_y*2 + btn_h), (col1_x + btn_w, cY + gap_y*2 + btn_h*2), (0, 100, 200), -1)
            cv2.putText(frame, "DANG KY (FILE)", (col1_x + int(50 * (w/800)), cY + gap_y*2 + btn_h + int(42 * (h/600))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)
        
        # Column 2
        # Button 4: Edit (All)
        cv2.rectangle(frame, (col2_x, cY - btn_h - gap_y), (col2_x + btn_w, cY - gap_y), (100, 100, 100), -1)
        cv2.putText(frame, "CHINH SUA", (col2_x + int(75 * (w/800)), cY - gap_y - int(18 * (h/600))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)

        # Button 5: List (All)
        cv2.rectangle(frame, (col2_x, cY + gap_y), (col2_x + btn_w, cY + gap_y + btn_h), (150, 50, 150), -1)
        cv2.putText(frame, "DANH SACH", (col2_x + int(75 * (w/800)), cY + gap_y + int(42 * (h/600))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * (w/800), (255, 255, 255), 2)

        # Button 6: History
        cv2.rectangle(frame, (col2_x, cY + gap_y*2 + btn_h), (col2_x + btn_w, cY + gap_y*2 + btn_h*2), (100, 50, 0), -1)
        cv2.putText(frame, "LICH SU", (col2_x + int(90 * (w/800)), cY + gap_y*2 + btn_h + int(42 * (h/600))),
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
        
        # Bottom Management Buttons (Admin Only)
        if str(self.session_role).lower() == 'admin':
            # 1. Connect HKB (Left)
            cv2.putText(frame, "KET NOI HKB", (cX - 355, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 2. Manage Users (Center)
            cv2.rectangle(frame, (cX - 120, h - 84), (cX + 120, h - 20), (60, 60, 180), -1)
            cv2.putText(frame, "QUAN LY USER", (cX - 95, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 3. Manage Company (Right)
            cv2.rectangle(frame, (cX + 140, h - 84), (cX + 380, h - 20), (100, 50, 150), -1)
            cv2.putText(frame, "QUAN LY CONG TY", (cX + 160, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 4. Settings Button (Top Left)
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
        
        return frame

    @staticmethod
    def get_user_form(include_upload=False, session_role=None, mongo_db=None):
        """
        Opens a centered tkinter dialog to collect User ID, Name, Birthday, Company, and optionally Photos.
        Returns: (ID, Name, Birthday, file_paths, company_id) or None if cancelled.
        """
        import tkinter as tk
        from tkinter import messagebox, filedialog

        root = tk.Tk()
        root.title("Form Đăng Ký Người Dùng")
        
        # Center the window
        # Dynamic height based on fields
        base_h = 300
        if include_upload: base_h += 100
        if session_role == 'admin': base_h += 60
        
        window_width, window_height = 380, base_h
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        root.attributes('-topmost', True)
        root.focus_force()

        form_data = {"id": None, "name": None, "bday": None, "files": [], "company_id": None}
        
        def on_select_files():
            paths = filedialog.askopenfilenames(
                title="Chọn ảnh khuôn mặt (Tối đa 5)",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if paths:
                form_data["files"] = list(paths)[:5]
                lbl_file_count.config(text=f"Đã chọn: {len(form_data['files'])} ảnh")

        def on_submit():
            u_id = entry_id.get().strip()
            u_name = entry_name.get().strip()
            u_bday = entry_bday.get().strip()
            
            # Get company_id based on role or selection
            if str(session_role).lower() == 'admin':
                selected_cid = combo_cid.get()
                if not selected_cid:
                    messagebox.showwarning("!", "Vui lòng chọn Công ty")
                    return
                form_data["company_id"] = selected_cid
            else:
                # Default for non-admin will be handled in main or passed in session
                form_data["company_id"] = None 

            if not u_id or not u_name:
                messagebox.showwarning("Cảnh báo", "Vui lòng nhập ID và Họ tên!")
                return
            
            if include_upload and not form_data["files"]:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn ít nhất 1 ảnh!")
                return
            
            form_data["id"] = u_id
            form_data["name"] = u_name
            form_data["bday"] = u_bday or "N/A"
            root.destroy()

        # UI Elements
        tk.Label(root, text="ĐĂNG KÝ THÔNG TIN", font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Label(root, text="Mã nhân viên *:").pack()
        entry_id = tk.Entry(root, width=30)
        entry_id.pack(pady=2)
        entry_id.focus_set()

        tk.Label(root, text="Họ và tên *:").pack()
        entry_name = tk.Entry(root, width=30)
        entry_name.pack(pady=2)

        tk.Label(root, text="Ngày sinh (DD/MM/YYYY):").pack()
        entry_bday = tk.Entry(root, width=30)
        entry_bday.pack(pady=2)

        if str(session_role).lower() == 'admin' and mongo_db:
            from tkinter import ttk
            tk.Label(root, text="Phân quyền Công ty *:").pack(pady=(5, 0))
            companies = mongo_db.get_all_companies()
            company_list = [c.get("company_id") for c in companies]
            if "admin" not in company_list: company_list = ["admin"] + company_list
            
            combo_cid = ttk.Combobox(root, values=company_list, width=27)
            combo_cid.set("admin")
            combo_cid.pack(pady=2)

        if include_upload:
            tk.Label(root, text="Ảnh khuôn mặt *:").pack(pady=(10, 0))
            tk.Button(root, text="Chọn ảnh", command=on_select_files).pack(pady=2)
            lbl_file_count = tk.Label(root, text="Chưa chọn ảnh", fg="gray")
            lbl_file_count.pack()

        tk.Button(root, text="XÁC NHẬN", command=on_submit, width=15, bg="#28a745", fg="white").pack(pady=15)
        
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()
        
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
        root.withdraw()
        u_id = simpledialog.askstring("Chỉnh sửa", "Nhập Mã nhân viên cần chỉnh sửa:", parent=root)
        root.destroy()
        return u_id

    @staticmethod
    def get_edit_user_form(current_id, current_name, current_bday):
        """
        Dialog to edit Name and Birthday.
        """
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.title("Chỉnh sửa thông tin")
        
        window_width, window_height = 350, 280
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        root.attributes('-topmost', True)
        root.focus_force()

        result = {"name": None, "bday": None, "delete": False}
        
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

        tk.Label(root, text=f"ID: {current_id}", font=("Arial", 10, "bold")).pack(pady=10)
        
        tk.Label(root, text="Họ và tên:").pack()
        entry_name = tk.Entry(root, width=30)
        entry_name.insert(0, current_name)
        entry_name.pack(pady=2)

        tk.Label(root, text="Ngày sinh (DD/MM/YYYY):").pack()
        entry_bday = tk.Entry(root, width=30)
        entry_bday.insert(0, current_bday)
        entry_bday.pack(pady=2)

        tk.Button(root, text="LƯU THAY ĐỔI", command=on_save, width=20, bg="#28a745", fg="white").pack(pady=10)
        tk.Button(root, text="XÓA NHÂN VIÊN", command=on_delete, width=20, bg="#dc3545", fg="white").pack(pady=5)
        
        root.mainloop()
        return result if result["name"] or result["delete"] else None

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
        columns = ("id", "name", "birthday")
        tree = ttk.Treeview(root, columns=columns, show="headings")
        
        tree.heading("id", text="Mã nhân viên")
        tree.heading("name", text="Họ và tên")
        tree.heading("birthday", text="Ngày sinh")
        
        tree.column("id", width=120)
        tree.column("name", width=200)
        tree.column("birthday", width=150)

        for user in user_list:
            tree.insert("", tk.END, values=(user["user_id"], user["user_name"], user["birthday"]))

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
        root.geometry("500x400")
        root.attributes('-topmost', True)

        tk.Label(root, text="CHỌN NHÂN VIÊN CẦN CHỈNH SỬA", font=("Arial", 11, "bold")).pack(pady=10)

        selected = {"id": None}
        
        tree = ttk.Treeview(root, columns=("id", "name", "bday"), show="headings")
        tree.heading("id", text="Mã NV")
        tree.heading("name", text="Họ tên")
        tree.heading("bday", text="Ngày sinh")
        
        tree.column("id", width=100)
        tree.column("name", width=200)
        tree.column("bday", width=120)

        for u in user_list:
            tree.insert("", tk.END, values=(u.get('user_id'), u.get('user_name'), u.get('birthday')))

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
    def show_attendance_logs_ui(logs):
        """
        Displays a table of attendance logs using tkinter.
        """
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("Lịch sử điểm danh (Hôm nay)")
        
        window_width, window_height = 700, 500
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        root.attributes('-topmost', True)

        label = tk.Label(root, text=f"Lịch sử điểm danh hôm nay ({len(logs)} lượt)", font=("Arial", 11, "bold"))
        label.pack(pady=10)

        # Create Treeview
        columns = ("id", "user_id", "name", "time", "status")
        tree = ttk.Treeview(root, columns=columns, show="headings")
        
        tree.heading("id", text="ID")
        tree.heading("user_id", text="Mã NV")
        tree.heading("name", text="Họ và tên")
        tree.heading("time", text="Thời gian")
        tree.heading("status", text="Trạng thái")
        
        tree.column("id", width=50)
        tree.column("user_id", width=100)
        tree.column("name", width=180)
        tree.column("time", width=180)
        tree.column("status", width=100)

        # logs structure from MongoDB: (id, user_id, user_name, timestamp, date, status, image_path)
        for log in logs:
            # Note: status is at index 5 now
            status_val = log[5] if len(log) > 5 else "N/A"
            tree.insert("", tk.END, values=(log[0], log[1], log[2], log[3], status_val))

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        def on_upload():
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn một lượt điểm danh để upload!")
                return
            
            from src.services.hkb_service import hkb_service
            from tkinter import messagebox
            
            item_values = tree.item(selected_item)['values']
            # item_values: (id, user_id, name, time, status)
            log_id = str(item_values[0])
            log_data = next((l for l in logs if str(l[0]) == log_id), None)
            
            if log_data:
                res = hkb_service.upload_attendance(
                    user_id=log_data[1],
                    user_name=log_data[2],
                    timestamp=log_data[3],
                    status=log_data[5],
                    image_path=log_data[6]
                )
                if res:
                    messagebox.showinfo("Thành công", f"Đã upload lượt điểm danh của {log_data[2]}")
                else:
                    messagebox.showerror("Lỗi", "Upload thất bại. Vui lòng kiểm tra cấu hình HKB.")

        def on_view_image():
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("Cảnh báo", "Vui lòng chọn một lượt điểm danh để xem ảnh!")
                return
            
            from src.attendance.mongodb_mgr import mongo_db
            import cv2
            import numpy as np
            from tkinter import messagebox

            item_values = tree.item(selected_item)['values']
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
        tk.Button(btn_container, text="UPLOAD LÊN HKB", command=on_upload, width=15, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)
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
            
            # Truyền user_id hiện tại để lọc danh sách UUID đã đăng ký
            connections = hkb_service.get_connections(user_id=current_user_id)
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

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        btn_container = tk.Frame(root)
        btn_container.pack(pady=10)

        tk.Button(btn_container, text="LÀM MỚI", command=refresh_list, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐĂNG KÝ HỆ THỐNG", command=on_register, width=15, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="HỦY KẾT NỐI", command=on_revoke, width=15, bg="#dc3545", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐÓNG", command=root.destroy, width=15, bg="#007bff", fg="white").pack(side=tk.LEFT, padx=5)

        refresh_list()
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
        
        def refresh():
            for i in tree.get_children(): tree.delete(i)
            for u in mongo_db.get_all_cloud_users():
                tree.insert("", tk.END, values=(u.get('username'), u.get('role'), u.get('company_id')))

        def on_add():
            from tkinter import ttk
            add_win = tk.Toplevel(root); add_win.title("Thêm User Mới"); add_win.geometry("300x400")
            
            tk.Label(add_win, text="Username:").pack(pady=5)
            e_user = tk.Entry(add_win); e_user.pack()
            
            tk.Label(add_win, text="Password:").pack(pady=5)
            e_pwd = tk.Entry(add_win, show="*"); e_pwd.pack()
            
            tk.Label(add_win, text="Quyền hạn:").pack(pady=5)
            e_role = ttk.Combobox(add_win, values=["admin", "company", "user"])
            e_role.set("company"); e_role.pack()
            
            tk.Label(add_win, text="Phân quyền Công ty:").pack(pady=5)
            # Fetch all companies to show in dropdown
            companies = mongo_db.get_all_companies()
            company_list = ["admin"] + [c.get("company_id") for c in companies]
            
            e_cid = ttk.Combobox(add_win, values=company_list)
            e_cid.set("admin"); e_cid.pack()
            
            def submit():
                u, p, r, cid = e_user.get(), e_pwd.get(), e_role.get(), e_cid.get()
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
    def show_system_settings_ui(mongo_db):
        """UI to manage system-wide settings like Group Keys."""
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.title("Cài đặt Hệ thống")
        root.geometry("500x350")
        root.attributes('-topmost', True)

        tk.Label(root, text="CẤU HÌNH HỆ THỐNG", font=("Arial", 14, "bold")).pack(pady=20)

        # 1. Group Keys Setting
        tk.Label(root, text="Group Keys (Cung cấp nhiều key, ngăn cách bởi dấu phẩy):", font=("Arial", 10)).pack(anchor="w", padx=20)
        
        # Lấy giá trị hiện tại từ DB
        current_keys = mongo_db.get_setting("group_keys", "")
        
        text_keys = tk.Text(root, height=5, width=55)
        text_keys.pack(pady=10, padx=20)
        text_keys.insert("1.0", current_keys)

        def save_settings():
            new_keys = text_keys.get("1.0", "end-1c").strip()
            if mongo_db.set_setting("group_keys", new_keys):
                messagebox.showinfo("Thành công", "Đã lưu cài đặt hệ thống!")
                root.destroy()
            else:
                messagebox.showerror("Lỗi", "Không thể lưu cài đặt!")

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="LƯU CÀI ĐẶT", command=save_settings, bg="#28a745", fg="white", width=15, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
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
