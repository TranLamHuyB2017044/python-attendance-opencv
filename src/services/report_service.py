import requests
import json
import os
from loguru import logger
from src.config import AuthServiceConfig, ReportSystemConfig
from src.attendance.mongodb_mgr import mongo_db
from src.services.hkb_service import hkb_service
from src.services.ping_service import _build_devices_info

class ReportService:
    """
    Service to report system logs and errors to the central monitoring system.
    """
    
    def report_error(self, message: str, status_code: int = 500, priority: str = "HIGH", image_url: str = None):
        """Convenience method to report an error log."""
        return self.report_log(message, status_code, log_type="ERROR", priority=priority, image_url=image_url)

    def report_info(self, message: str, status_code: int = 200, priority: str = "LOW", image_url: str = None):
        """Convenience method to report an info log."""
        return self.report_log(message, status_code, log_type="INFO", priority=priority, image_url=image_url)

    def report_log(self, message: str, status_code: int, log_type: str = "ERROR", priority: str = "MEDIUM", devices_info: dict = None, image_url: str = None):
        """
        Report a log entry to the central system.
        
        Steps:
        1. Save log to local MongoDB (system_logs collection).
        2. Authenticate via HKB Service to get the Bearer Token.
        3. Send the log to the Report System API.
        """
        try:
            # 1. Thu thập thông tin thiết bị nếu chưa có
            if devices_info is None:
                devices_info = _build_devices_info()

            # 2. Lưu log vào MongoDB local trước
            trace_id = mongo_db.save_system_log(
                message=message,
                status_code=status_code,
                log_type=log_type,
                priority=priority,
                devices_info=devices_info
            )
            
            if not trace_id:
                logger.error("ReportService: Failed to save log to MongoDB. Aborting API report.")
                return False

            # Đính kèm URL ảnh vào message nếu có
            full_message = message
            if image_url:
                full_message = f"{message}\n\n🖼️ Photo URL: {image_url}"

            # 3. Lấy Token từ HKB Service dành riêng cho Log System
            log_system_id = os.getenv("LOGX_SYSTEM_ID")
            log_api_key = os.getenv("LOGX_API_KEY")

            auth_result = hkb_service.authenticate(
                system_id=log_system_id,
                api_key=log_api_key,
                user_id=1
            )
            
            if not auth_result or not auth_result.success:
                reason = getattr(auth_result, 'message', 'Unknown reason')
                logger.error(f"ReportService: Authentication failed ({reason}). Cannot report log. TraceID: {trace_id}")
                return False
                
            # Trích xuất token từ response (giả định cấu trúc data.access_token hoặc token)
            token = None
            if hasattr(auth_result, 'data') and auth_result.data:
                token = auth_result.data.get('access_token') or auth_result.data.get('token')
            
            if not token:
                logger.error(f"ReportService: No token found in auth response. Data: {auth_result.data} | TraceID: {trace_id}")
                return False

            # 4. Gửi Log lên hệ thống trung tâm
            url = ReportSystemConfig.report_log_url()
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Trace-Id": str(trace_id),
                "Content-Type": "application/json"
            }
            
            payload = {
                "system_id": AuthServiceConfig.SYSTEM_ID,
                "message": full_message,
                "status_code": status_code,
                "type": log_type,
                "priority": priority,
                "devices_info": devices_info
            }

            logger.info(f"ReportService: 📡 Sending log report to {url} | TraceID: {trace_id} - Payload: {payload}")
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code in (200, 201):
                logger.success(f"ReportService: ✅ Log reported successfully. Server responded: {response.text}")
                return True
            else:
                logger.warning(f"ReportService: ⚠️ Failed to report log. Status: {response.status_code}, Response: {response.text}")
                return False

        except Exception as e:
            import traceback
            logger.error(f"ReportService: Critical error reporting log: {e}\n{traceback.format_exc()}")
            return False

# Singleton instance
report_service = ReportService()
