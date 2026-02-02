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

class AttendanceUI:
    """
    Handles all UI rendering and interaction logic.
    """
    def __init__(self):
        self.current_state = STATE_MENU

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
    def draw_status_bar(frame, fps, processing_time):
        """Draw FPS and AI stats."""
        cv2.putText(frame, f"FPS: {fps} | AI: {processing_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Nhan 'M' de ve Menu", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return frame
