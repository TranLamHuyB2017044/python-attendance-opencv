import ctypes
import os
import sys
from loguru import logger

# Constants for Windows API
ERROR_ALREADY_EXISTS = 183

def is_already_running(app_name):
    """
    Checks if another instance of the app is already running using a named Mutex.
    Works specifically for Windows.
    """
    if sys.platform != "win32":
        # Basic fallback for other platforms (simplified)
        return False

    mutex_name = f"Global\\Bittech_{app_name}_Mutex"
    
    # Create a named mutex
    kernel32 = ctypes.windll.kernel32
    mutex = kernel32.CreateMutexW(None, False, mutex_name)
    last_error = kernel32.GetLastError()

    if last_error == ERROR_ALREADY_EXISTS:
        # Mutex already exists, someone else is running
        return True
    
    # Keep the mutex handle alive for the duration of the process
    # We store it in a global or attribute to prevent garbage collection
    is_already_running._mutex_handle = mutex
    return False

def force_single_instance(app_name, exit_callback=None):
    """
    Ensures only one instance of the app is running.
    Exits the process if another instance is detected.
    """
    if is_already_running(app_name):
        msg = f"Ứng dụng {app_name} đang chạy! Không thể mở thêm cửa sổ mới."
        logger.warning(msg)
        
        # Try to show a visual message before exiting
        try:
            from src.utils.notification import show_error_message
            show_error_message("Lỗi khởi động", f"Một bản sao của {app_name} đã được mở trước đó.\n\nVui lòng kiểm tra dưới thanh Taskbar hoặc Task Manager.")
        except:
            pass
            
        if exit_callback:
            exit_callback()
        sys.exit(0)
