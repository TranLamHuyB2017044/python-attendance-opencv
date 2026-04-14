"""
webhook_log_bus.py
──────────────────
Shared bus cho log panel:
  - SENT: nhận event khi tracker gửi webhook đi
  - GET:  nhận bản ghi khi API polling trả về record mới

Cả 2 luồng đều được lưu trong một deque thread-safe và phân biệt
bởi trường `log_type` ("SENT" | "GET").
"""
import threading
import datetime
import time
import json
import urllib.request
from collections import deque
from loguru import logger

# ── Configuration ────────────────────────────────────────────────────────────
_WEBHOOK_API_URL = "https://voice-cheking.bittechx.cloud/api/users"
_LOG_POLL_SEC    = 3.0

# ── Public state (read by panel renderer) ────────────────────────────────────
_lock        = threading.Lock()
_bus: deque  = deque(maxlen=200)   # newest last
_last_get_id = -1                  # highest id seen from GET
_poll_started = False

def start_polling():
    """Bắt đầu thread poll API lấy log GET."""
    global _poll_started
    with _lock:
        if _poll_started:
            return
        _poll_started = True
    
    def _worker():
        while True:
            try:
                req = urllib.request.Request(
                    _WEBHOOK_API_URL,
                    headers={"User-Agent": "BitTech-Monitor/2.0"}
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and data:
                    push_get(data)
            except Exception as exc:
                logger.debug(f"[LogBus] GET fetch error: {exc}")
            time.sleep(_LOG_POLL_SEC)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    logger.info(f"Webhook log polling started: {_WEBHOOK_API_URL}")

def push_sent(user_id: str, user_name: str, status: str,
              voice_text: str = "", time_str: str = ""):
    """Ghi 1 bản ghi SENT (tracker → webhook)."""
    if not time_str:
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {
        "log_type":  "SENT",
        "user_id":   user_id,
        "user_name": user_name,
        "status":    status,
        "voice_text": voice_text,
        "time":       time_str,
        "ts":         datetime.datetime.now().timestamp(),
        "is_new":     True,
    }
    with _lock:
        _bus.append(entry)

def push_get(records: list):
    """Cập nhật bản ghi GET từ API poll; chỉ thêm record chưa thấy."""
    global _last_get_id
    with _lock:
        old_max = _last_get_id
    new_max = old_max
    new_entries = []
    for r in records:
        eid = r.get("id", -1)
        if eid > old_max:
            entry = {
                "log_type":   "GET",
                "id":          eid,
                "user_id":     r.get("user_id", ""),
                "user_name":   r.get("user_name", ""),
                "status":      r.get("status", "unknown"),
                "voice_text":  r.get("voice_text", ""),
                "time":        r.get("time", ""),
                "ts":          datetime.datetime.now().timestamp(),
                "is_new":      True,
            }
            new_entries.append(entry)
            new_max = max(new_max, eid)
    if new_entries:
        with _lock:
            for e in reversed(new_entries):   # newest-last order
                _bus.append(e)
            _last_get_id = new_max

def get_entries(limit: int = 60):
    """Trả danh sách newest-first (reverse deque) để vẽ từ trên xuống."""
    with _lock:
        entries = list(_bus)
    # Mark is_new = False after first read (panel diff highlight)
    entries.reverse()                   # newest first
    return entries[:limit]

def mark_all_read():
    """Tắt is_new sau khi render đã xong 1 vòng."""
    with _lock:
        for e in _bus:
            e["is_new"] = False

def clear_logs():
    """Xóa sạch bảng log hiện tại."""
    with _lock:
        _bus.clear()
    logger.info("Webhook log panel cleared.")
