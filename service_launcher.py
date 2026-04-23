import os
import sys
import urllib.request
import json
import zipfile
import shutil
import urllib.parse
from datetime import datetime

# ==========================================
# CẤU HÌNH AUTO UPDATER NGẦM
# ==========================================
GITHUB_REPO = "TranLamHuyB2017044/python-attendance-opencv" 
GITHUB_TOKEN = "" # Bắt buộc điền nếu Repo Private
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

SRC_DIR = "src"
BACKUP_DIR = "src_backup"
VERSION_FILE = "local_version.txt"
ZIP_NAME = "src_update.zip"

# Đọc cấu hình từ file .env (Để lấy Token Telegram)
def load_env():
    env_vars = {}
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, val = line.strip().split('=', 1)
                    env_vars[key.strip()] = val.strip()
    return env_vars

env_config = load_env()
GITHUB_TOKEN = env_config.get("GITHUB_TOKEN", "") # Lấy token từ .env
TELEGRAM_BOT_TOKEN = env_config.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = env_config.get("TELEGRAM_CHAT_ID", "")

# --- CÁC HÀM TIỆN ÍCH ---
def write_log(message):
    """Ghi log hệ thống."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    with open("updater.log", "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())

def send_telegram(message):
    """Gửi thông báo Telegram (Sử dụng HTTP cơ bản để không phụ thuộc requests)."""
    write_log(message) # Ghi vào log local
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({'chat_id': TELEGRAM_CHAT_ID, 'text': message}).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=5) as response:
            pass
    except Exception as e:
        write_log(f"Lỗi gửi Telegram: {e}")

    # --- SONG SONG: Gửi log lên Dashboard trung tâm ---
    try:
        # Thêm đường dẫn hiện tại vào sys.path để tìm thấy src.services
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from src.services.report_service import report_service
        
        status_code = 200 # Mặc định Info/Success
        log_type = "INFO"
        if "❌" in message or "Lỗi" in message:
            status_code = 500
            log_type = "ERROR"
            
        report_service.report_log(
            message=f"[Update Service] {message}",
            status_code=status_code,
            log_type=log_type
        )
    except Exception:
        # Không log lỗi này để tránh loop hoặc rác updater.log nếu src/ chưa sẵn sàng
        pass

def get_headers():
    headers = {'User-Agent': 'Attendance-Updater'}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    return headers

def get_local_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "0.0.0"

# --- LOGIC CẬP NHẬT CHÍNH ---
def check_and_update_silently():
    local_ver = get_local_version()
    try:
        req = urllib.request.Request(GITHUB_API_URL, headers=get_headers())
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            latest_ver = data.get('tag_name', 'v0.0.0').replace('v', '')
            assets = data.get('assets', [])
            
            download_url = None
            for asset in assets:
                if asset['name'] == ZIP_NAME:
                    download_url = asset['url'] if GITHUB_TOKEN else asset['browser_download_url']
                    break
    except Exception as e:
        write_log(f"Không thể kết nối GitHub để check update: {e}")
        return False

    def is_newer(ver1, ver2):
        if not ver1 or not ver2: return False
        try:
            return [int(x) for x in ver1.split('.')] > [int(x) for x in ver2.split('.')]
        except: return False

    # NẾU CÓ BẢN MỚI
    if latest_ver and download_url and is_newer(latest_ver, local_ver):
        send_telegram(f"⏳ Phát hiện phiên bản mới (v{latest_ver}). Đang tiến hành tải cập nhật ngầm...")
        update_temp_zip = "temp_update.zip"
        
        # 1. Tải file ZIP
        try:
            headers = get_headers()
            if GITHUB_TOKEN:
                headers['Accept'] = 'application/octet-stream'
            req = urllib.request.Request(download_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(update_temp_zip, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            send_telegram(f"❌ Lỗi khi tải bản cập nhật v{latest_ver}: {e}")
            return False

        # 2. Xóa src_backup cũ và Backup src hiện tại
        try:
            if os.path.exists(BACKUP_DIR):
                shutil.rmtree(BACKUP_DIR)
            if os.path.exists(SRC_DIR):
                shutil.move(SRC_DIR, BACKUP_DIR)
        except Exception as e:
            send_telegram(f"❌ Lỗi khi backup thư mục mã nguồn: {e}")
            return False

        # 3. Giải nén ghi đè
        try:
            with zipfile.ZipFile(update_temp_zip, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Lưu version mới
            with open(VERSION_FILE, 'w', encoding='utf-8') as f:
                f.write(latest_ver)
                
            send_telegram(f"✅ Cập nhật thành công phần mềm lên phiên bản v{latest_ver}!")
            
            if os.path.exists(update_temp_zip):
                os.remove(update_temp_zip)
                
            return True
            
        except Exception as e:
            # ROLLBACK
            if os.path.exists(SRC_DIR):
                shutil.rmtree(SRC_DIR)
            if os.path.exists(BACKUP_DIR):
                shutil.move(BACKUP_DIR, SRC_DIR)
            send_telegram(f"❌ Giải nén thất bại. Đã khôi phục lại phiên bản cũ (v{local_ver}). Chi tiết lỗi: {e}")
            return False
            
    return False # Không có update hoặc là phiên bản mới nhất r

# ==========================================
# KHỞI ĐỘNG CHƯƠNG TRÌNH ATTENDANCE SERVICE
# ==========================================
if __name__ == "__main__":
    # BƯỚC 1: TẢI BẢN CẬP NHẬT (NẾU CÓ)
    check_and_update_silently()
    
    # BƯỚC 2: ƯU TIÊN MÃ NGUỒN TỪ THƯ MỤC BÊN NGOÀI
    # Trick cực quan trọng của Loader Flow: Đẩy thư mục hiện tại lên đầu sys.path
    # Để chắc chắn Python sẽ load file từ thư mục /src/ tải về chứ không lấy từ _internal
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        if exe_dir not in sys.path:
            sys.path.insert(0, exe_dir)
            
    # BƯỚC 3: TIẾN HÀNH KHỞI CHẠY APP THỰC TẾ
    write_log("Đang nạp mã nguồn hệ thống (src/)...")
    
    try:
        from src.service_main import main
        main()
    except Exception as e:
        msg = f"❌ Lỗi nghiêm trọng (Crash) khi khởi động Attendance Service: {e}"
        send_telegram(msg)
        sys.exit(1)
