import customtkinter as ctk
import tkinter as tk
from datetime import datetime
import calendar

class CTkDatePicker(ctk.CTkToplevel):
    def __init__(self, parent=None, title="Select Date", initial_date=None, ok_button_text=None, **kwargs):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x380") if ok_button_text else self.geometry("300x340")
        self.attributes('-topmost', True)
        self.resizable(False, False)
        
        # Appearance
        self.configure(fg_color="#1a1a1a")
        
        # Center the window
        if parent:
            self.transient(parent)
            self.update_idletasks()
            w, h = (300, 380) if ok_button_text else (300, 340)
            px = parent.winfo_x() + (parent.winfo_width() // 2) - (w // 2)
            py = parent.winfo_y() + (parent.winfo_height() // 2) - (h // 2)
            self.geometry(f"{w}x{h}+{px}+{py}")
        
        try:
            if initial_date:
                if "-" in initial_date:
                    self.current_date = datetime.strptime(initial_date, "%Y-%m-%d")
                else: # Handle DD-MM-YYYY
                    self.current_date = datetime.strptime(initial_date, "%d-%m-%Y")
            else:
                self.current_date = datetime.now()
        except:
            self.current_date = datetime.now()
            
        self.selected_date = None
        self.ok_button_text = ok_button_text
        self.current_month = self.current_date.month
        self.current_year = self.current_date.year
        
        self.setup_ui()
        self.draw_calendar()
        self.grab_set()

    def setup_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        self.btn_prev = ctk.CTkButton(header, text="<", width=35, height=35, 
                                     fg_color="#1f6aa5", hover_color="#154c75",
                                     command=self.prev_month)
        self.btn_prev.pack(side="left")
        
        self.lbl_month = ctk.CTkLabel(header, text="", font=("Arial", 16, "bold"))
        self.lbl_month.pack(side="left", expand=True)
        
        self.btn_next = ctk.CTkButton(header, text=">", width=35, height=35, 
                                     fg_color="#1f6aa5", hover_color="#154c75",
                                     command=self.next_month)
        self.btn_next.pack(side="right")
        
        # Day names header
        days_header = ctk.CTkFrame(self, fg_color="transparent")
        days_header.pack(fill="x", padx=10)
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for d in days:
            ctk.CTkLabel(days_header, text=d, width=38, font=("Arial", 11, "bold"), 
                        text_color="gray").pack(side="left", padx=1)
            
        # Calendar Grid
        self.grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.grid_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Confirm Button if needed
        if self.ok_button_text:
            btn_confirm = ctk.CTkButton(self, text=self.ok_button_text, height=40,
                                      fg_color="#28a745", hover_color="#218838",
                                      font=("Arial", 13, "bold"),
                                      command=self.confirm_selection)
            btn_confirm.pack(fill="x", padx=20, pady=(5, 20))

    def draw_calendar(self):
        # Clear grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
            
        month_name = calendar.month_name[self.current_month]
        self.lbl_month.configure(text=f"{month_name}, {self.current_year}")
        
        cal = calendar.monthcalendar(self.current_year, self.current_month)
        today = datetime.now()
        
        for r, week in enumerate(cal):
            for c, day in enumerate(week):
                if day == 0:
                    ctk.CTkLabel(self.grid_frame, text="", width=38, height=38).grid(row=r, column=c, padx=1, pady=1)
                else:
                    is_today = (day == today.day and self.current_month == today.month and self.current_year == today.year)
                    
                    # Highlight selection
                    is_selected = False
                    if self.selected_date:
                        sel_y, sel_m, sel_d = map(int, self.selected_date.split("-"))
                        is_selected = (day == sel_d and self.current_month == sel_m and self.current_year == sel_y)
                    elif not self.selected_date:
                        # Fallback to current_date for initial highlight
                        is_selected = (day == self.current_date.day and self.current_month == self.current_date.month and self.current_year == self.current_date.year)
                    
                    fg = "#1f6aa5" if is_selected else "transparent"
                    text_color = "white"
                    border = 1 if is_today and not is_selected else 0
                    
                    btn = ctk.CTkButton(self.grid_frame, text=str(day), width=36, height=36, 
                                      fg_color=fg, 
                                      text_color=text_color,
                                      border_width=border, border_color="#1f6aa5",
                                      corner_radius=8,
                                      hover_color="#3a7ebf",
                                      font=("Arial", 12),
                                      command=lambda d=day: self.select_date(d))
                    btn.grid(row=r, column=c, padx=1, pady=1)

    def prev_month(self):
        self.current_month -= 1
        if self.current_month < 1:
            self.current_month = 12
            self.current_year -= 1
        self.draw_calendar()

    def next_month(self):
        self.current_month += 1
        if self.current_month > 12:
            self.current_month = 1
            self.current_year += 1
        self.draw_calendar()

    def select_date(self, day):
        self.selected_date = f"{self.current_year}-{self.current_month:02d}-{day:02d}"
        if not self.ok_button_text:
            self.destroy()
        else:
            self.draw_calendar()

    def confirm_selection(self):
        if not self.selected_date:
            # If nothing selected, use the current visible date as default
            self.selected_date = f"{self.current_year}-{self.current_month:02d}-{self.current_date.day:02d}"
        self.destroy()

    def get_date(self):
        self.wait_window()
        return self.selected_date

class CTkDateEntry(ctk.CTkFrame):
    def __init__(self, master, width=200, height=35, placeholder="", initial_date=None, **kwargs):
        super().__init__(master, fg_color="transparent", width=width, height=height)
        
        self.grid_columnconfigure(0, weight=1)
        
        self.entry = ctk.CTkEntry(self, placeholder_text=placeholder, height=height)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        if initial_date:
            self.entry.insert(0, initial_date)
            
        self.btn = ctk.CTkButton(self, text="▼", width=35, height=height, 
                                fg_color="#1f6aa5", hover_color="#154c75",
                                command=self.open_picker)
        self.btn.grid(row=0, column=1)
        
    def open_picker(self):
        current_val = self.entry.get()
        picker = CTkDatePicker(parent=self.winfo_toplevel(), initial_date=current_val)
        selected = picker.get_date()
        
        if selected:
            # Convert YYYY-MM-DD to DD-MM-YYYY for the entry if needed, 
            # or keep as is. Usually internal apps use YYYY-MM-DD or DD-MM-YYYY.
            # Based on app_ui.py usage, it seems to prefer DD-MM-YYYY for display.
            try:
                date_obj = datetime.strptime(selected, "%Y-%m-%d")
                formatted = date_obj.strftime("%d-%m-%Y")
                self.entry.delete(0, tk.END)
                self.entry.insert(0, formatted)
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(0, selected)
                
    def get(self):
        return self.entry.get()
    
    def set(self, value):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, value)
