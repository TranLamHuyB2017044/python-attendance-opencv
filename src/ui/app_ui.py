import cv2
import numpy as np
from loguru import logger

# State Management Constants
STATE_MENU = 0
STATE_DETECT = 1
STATE_ENROLL_CAM = 2
STATE_ENROLL_UPLOAD = 3
STATE_EDIT = 4
STATE_LIST = 5
STATE_HISTORY = 6
STATE_HKB_LIST = 7

class AttendanceUI:
    """
    Handles all UI rendering and interaction logic.
    """
    def __init__(self):
        self.current_state = STATE_MENU
        self.is_admin_logged_in = False

    def handle_menu_click(self, event, x, y, flags, param):
        """Handle mouse clicks for the menu."""
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = param
            cX, cY = w // 2, h // 2
            
            # Column 1 (Left)
            col1_L, col1_R = cX - 310, cX - 10
            # Column 2 (Right)
            col2_L, col2_R = cX + 10, cX + 310

            # BAT DAU (Col 1, Row 1)
            if col1_L < x < col1_R and cY-80 < y < cY-20: 
                self.current_state = STATE_DETECT
                logger.info("UI: Switched to Detection Mode")
                
            # DANG KY CAM (Col 1, Row 2)
            elif col1_L < x < col1_R and cY+20 < y < cY+80: 
                self.current_state = STATE_ENROLL_CAM
                logger.info("UI: Switched to Enrollment (Camera) Mode")
                
            # DANG KY FILE (Col 1, Row 3)
            elif col1_L < x < col1_R and cY+120 < y < cY+180: 
                self.current_state = STATE_ENROLL_UPLOAD
                logger.info("UI: Switched to Enrollment (Upload) Mode")

            # CHINH SUA (Col 2, Row 1)
            elif col2_L < x < col2_R and cY-80 < y < cY-20: 
                self.current_state = STATE_EDIT
                logger.info("UI: Switched to Edit Mode")

            # DANH SACH (Col 2, Row 2)
            elif col2_L < x < col2_R and cY+20 < y < cY+80: 
                self.current_state = STATE_LIST
                logger.info("UI: Switched to List Mode")

            # LICH SU (Col 2, Row 3)
            elif col2_L < x < col2_R and cY+120 < y < cY+180: 
                self.current_state = STATE_HISTORY
                logger.info("UI: Switched to History Mode")

            # KET NOI HKB (Bottom Center)
            elif cX - 150 < x < cX + 150 and h - 80 < y < h - 20:
                self.current_state = STATE_HKB_LIST
                logger.info("UI: Switched to HKB Connection List")

    def draw_main_menu(self):
        """Draw a professional menu on a clean centered background."""
        w, h = 800, 600
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(frame, (0, 0), (w, h), (40, 40, 40), -1) 
        
        cX, cY = w // 2, h // 2
        
        # Title
        cv2.putText(frame, "HE THONG DIEM DANH AI", (cX - 240, cY - 150),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        # Draw Columns Layout
        col1_x = cX - 310
        col2_x = cX + 10

        # --- Column 1 ---
        # Button 1: Start System
        cv2.rectangle(frame, (col1_x, cY - 80), (col1_x + 300, cY - 20), (40, 180, 40), -1)
        cv2.putText(frame, "BAT DAU", (col1_x + 90, cY - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Button 2: Enroll Camera
        cv2.rectangle(frame, (col1_x, cY + 20), (col1_x + 300, cY + 80), (200, 120, 0), -1)
        cv2.putText(frame, "DANG KY (CAM)", (col1_x + 50, cY + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Button 3: Enroll Upload
        cv2.rectangle(frame, (col1_x, cY + 120), (col1_x + 300, cY + 180), (0, 100, 200), -1)
        cv2.putText(frame, "DANG KY (FILE)", (col1_x + 50, cY + 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # --- Column 2 ---
        # Button 4: Edit
        cv2.rectangle(frame, (col2_x, cY - 80), (col2_x + 300, cY - 20), (100, 100, 100), -1)
        cv2.putText(frame, "CHINH SUA", (col2_x + 75, cY - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Button 5: List
        cv2.rectangle(frame, (col2_x, cY + 20), (col2_x + 300, cY + 80), (150, 50, 150), -1)
        cv2.putText(frame, "DANH SACH", (col2_x + 75, cY + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Button 6: History
        cv2.rectangle(frame, (col2_x, cY + 120), (col2_x + 300, cY + 180), (100, 50, 0), -1)
        cv2.putText(frame, "LICH SU", (col2_x + 90, cY + 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # --- Instruction Table (Compact) ---
        table_x, table_y = 30, 480
        cv2.rectangle(frame, (table_x, table_y), (table_x + 200, table_y + 90), (60, 60, 60), -1)
        cv2.rectangle(frame, (table_x, table_y), (table_x + 200, table_y + 90), (100, 100, 100), 1)
        
        cv2.putText(frame, "PHIM TAT:", (table_x + 10, table_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        instructions = ["Q: Thoat", "M: Menu", "S: Chup anh", "C: Huy bỏ"]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (table_x + 10, table_y + 40 + (i * 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(frame, "Phat trien boi Biitech", (w - 180, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # New Button: Connect HKB
        cv2.rectangle(frame, (cX - 150, h - 80), (cX + 150, h - 20), (0, 165, 255), -1)
        cv2.putText(frame, "KET NOI HKB", (cX - 65, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    @staticmethod
    def get_user_form(include_upload=False):
        """
        Opens a centered tkinter dialog to collect User ID, Name, Birthday, and optionally Photos.
        Returns: (ID, Name, Birthday, file_paths) or None if cancelled.
        """
        import tkinter as tk
        from tkinter import messagebox, filedialog

        root = tk.Tk()
        root.title("Form Đăng Ký Người Dùng")
        
        # Center the window
        window_width, window_height = 350, 350 if include_upload else 250
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        pos_x = (screen_width // 2) - (window_width // 2)
        pos_y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        root.attributes('-topmost', True)
        root.focus_force()

        form_data = {"id": None, "name": None, "bday": None, "files": []}
        
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
            
        return form_data["id"], form_data["name"], form_data["bday"], form_data["files"]

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

        # logs structure from sqlite: (id, user_id, user_name, timestamp, date, status, image_path)
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
            # Find the original log to get the image_path
            log_id = item_values[0]
            log_data = next((l for l in logs if l[0] == log_id), None)
            
            if log_data:
                # log_data: (id, user_id, user_name, timestamp, date, status, image_path)
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

        btn_container = tk.Frame(root)
        btn_container.pack(pady=10)

        tk.Button(btn_container, text="UPLOAD LÊN HKB", command=on_upload, width=15, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐÓNG", command=root.destroy, width=15, bg="#007bff", fg="white").pack(side=tk.LEFT, padx=5)

        root.mainloop()

    @staticmethod
    def show_hkb_connections_ui():
        """
        Displays a list of HKB connections and allows registering/connecting.
        """
        import tkinter as tk
        from tkinter import ttk, messagebox
        from src.services.hkb_service import hkb_service
        from src.config import AuthServiceConfig

        root = tk.Tk()
        root.title("Kết nối HKB Auth")
        root.geometry("600x450")
        root.attributes('-topmost', True)

        tk.Label(root, text="DANH SÁCH KẾT NỐI HKB", font=("Arial", 12, "bold")).pack(pady=10)

        # Create Treeview
        columns = ("id", "system_id", "register", "description", "status")
        tree = ttk.Treeview(root, columns=columns, show="headings")
        
        tree.heading("id", text="ID")
        tree.heading("system_id", text="System ID")
        tree.heading("register", text="Module")
        tree.heading("description", text="Mô tả")
        tree.heading("status", text="Trạng thái")
        
        tree.column("id", width=50)
        tree.column("system_id", width=120)
        tree.column("register", width=100)
        tree.column("description", width=150)
        tree.column("status", width=80)

        def refresh_list():
            for item in tree.get_children():
                tree.delete(item)
            
            connections = hkb_service.get_connections()
            if connections and isinstance(connections, list):
                for conn in connections:
                    # Adjust based on actual API result structure
                    tree.insert("", tk.END, values=(
                        conn.get("id", "N/A"),
                        conn.get("system_id", "N/A"),
                        conn.get("system_register", "N/A"),
                        conn.get("description", "N/A"),
                        "Connected" # If it's in the list, assume it works
                    ))
            elif connections:
                # Might be a single dict or other structure
                logger.info(f"API result is not a list: {connections}")
            else:
                messagebox.showinfo("Thông báo", "Không tìm thấy kết nối nào hoặc lỗi API.")

        def on_register():
            # Simple dialog to register current system
            from tkinter import simpledialog
            desc = simpledialog.askstring("Đăng ký", "Nhập mô tả cho hệ thống này:", parent=root)
            if desc:
                # In a real app, external_id might be the device ID or similar
                result = hkb_service.register_client(
                    system_id=AuthServiceConfig.SYSTEM_ID,
                    external_id=1, # Default
                    description=desc,
                    user_info={"app": "Face Attendance System"}
                )
                if result:
                    messagebox.showinfo("Thành công", "Đã đăng ký hệ thống lên HKB.")
                    refresh_list()
                else:
                    messagebox.showerror("Lỗi", "Đăng ký thất bại.")

        tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        btn_container = tk.Frame(root)
        btn_container.pack(pady=10)

        tk.Button(btn_container, text="LÀM MỚI", command=refresh_list, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐĂNG KÝ HỆ THỐNG", command=on_register, width=15, bg="#28a745", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_container, text="ĐÓNG", command=root.destroy, width=15, bg="#007bff", fg="white").pack(side=tk.LEFT, padx=5)

        refresh_list()
        root.mainloop()

    def show_login_dialog(self):
        """
        Shows a login dialog for admin authentication.
        """
        import tkinter as tk
        from tkinter import messagebox
        from src.attendance.attendance_db import db as sqlite_db

        login_root = tk.Tk()
        login_root.title("Admin Login")
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
            if sqlite_db.verify_admin(user, pwd):
                login_status["authenticated"] = True
                self.is_admin_logged_in = True
                login_root.destroy()
            else:
                messagebox.showerror("Lỗi", "Sai tài khoản hoặc mật khẩu!")

        tk.Label(login_root, text="YÊU CẦU ĐĂNG NHẬP", font=("Arial", 10, "bold")).pack(pady=10)
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
    def draw_status_bar(frame, fps, processing_time):
        """Draw FPS and AI stats."""
        cv2.putText(frame, f"FPS: {fps} | AI: {processing_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Nhan 'M' de ve Menu", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return frame
