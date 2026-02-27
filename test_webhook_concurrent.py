import time
import requests
import threading
from loguru import logger
import sys
import os

# Thêm đường dẫn project vào sys.path để có thể import từ src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import WebhookConfig

def send_test_webhook(i):
    user_id = f"TEST_USER_{i}"
    user_name = f"Test Nhan Vien {i}"
    
    payload = {
        "user_id": user_id,
        "user_name": user_name,
        "status": "IN",
        "voice_text": f"Xin chào {user_name}, kiểm tra hệ thống thành công",
        "time": time.strftime("%H:%M:%S")
    }
    
    logger.info(f"[{i}] Đang gửi dữ liệu...")
    start_time = time.time()
    try:
        response = requests.post(
            WebhookConfig.USER_WEBHOOK_URL,
            json=payload,
            timeout=10 # Chờ tối đa 10s cho mỗi request
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            logger.success(f"[{i}] THÀNH CÔNG ({elapsed:.2f}s) - {response.text[:50]}")
        else:
            logger.error(f"[{i}] THẤT BẠI ({response.status_code}) - {response.text[:50]}")
    except Exception as e:
        logger.error(f"[{i}] LỖI KẾT NỐI: {e}")

def run_test():
    url = WebhookConfig.USER_WEBHOOK_URL
    logger.info(f"BẮT ĐẦU TEST WEBHOOK: {url}")
    logger.info("Mục tiêu: Gửi 10 dữ liệu nhân viên cùng một thời điểm (Concurrency Test)\n")
    
    threads = []
    
    # Tạo 10 luồng gửi cùng lúc để giả lập 10 người lọt vào camera đồng thời
    for i in range(1, 11):
        t = threading.Thread(target=send_test_webhook, args=(i,))
        threads.append(t)
    
    # Kích hoạt 10 luồng bắt đầu chạy gần như cùng 1 mili-giây
    for t in threads:
        t.start()
        
    # Chờ tất cả chạy xong
    for t in threads:
        t.join()
        
    logger.info("\nHOÀN THÀNH TEST WEBHOOK!")

if __name__ == "__main__":
    run_test()
