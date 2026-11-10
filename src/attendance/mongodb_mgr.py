"""HTTP client for attendance storage operations."""
import json
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4
from urllib.parse import urlsplit

import cv2
import requests
from loguru import logger

from src.attendance_transport import OPERATIONS, decode_value, encode_value
from src.config import AttendanceBackendConfig, AuthServiceConfig, CameraConfig, MongoDbConfig


class AttendanceBackendError(RuntimeError):
    """A storage operation could not be completed through FastAPI."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class AttendanceBackendClient:
    def __getattr__(self, operation):
        if operation not in OPERATIONS:
            raise AttributeError(operation)

        def call(*args, **kwargs):
            return self._call(operation, *args, **kwargs)

        return call

    def __init__(self):
        self.session = None
        self.device = None
        self.camera_company_ids = ()
        self.integration_token = None
        self.outbox = None
        self.attendance_notification = None
        self.session_lock = Lock()
        self.default_company = MongoDbConfig.COMPANY_ID

    def _request(self, method, path, *, authorization=None, body=None):
        address = urlsplit(AttendanceBackendConfig.URL)
        if address.scheme != "https" and not (
            address.scheme == "http" and address.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise AttendanceBackendError("Remote attendance backend must use HTTPS")
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        try:
            response = requests.request(
                method, f"{AttendanceBackendConfig.URL}{path}", headers=headers,
                data=json.dumps(body, default=encode_value) if body is not None else None,
                timeout=(3, 25), allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                raise AttendanceBackendError("Attendance backend redirect rejected", response.status_code)
            response.raise_for_status()
            return json.loads(response.content, object_hook=decode_value)
        except requests.HTTPError as error:
            code = error.response.status_code
            raise AttendanceBackendError(f"Attendance backend rejected {path} (HTTP {code})", code) from None
        except (requests.RequestException, ValueError) as error:
            raise AttendanceBackendError(f"Attendance backend unavailable ({type(error).__name__})") from None

    def _authorization(self):
        if self.device:
            return "Device " + self.device["api_key"]
        if self.session:
            return "Bearer " + self.session["access_token"]
        if self.integration_token:
            return "Integration " + self.integration_token
        raise AttendanceBackendError("Sign in or provision this camera first", 401)

    def authenticated_request(self, method, path, body=None):
        authorization = self._authorization()
        try:
            return self._request(method, path, authorization=authorization, body=body)
        except AttendanceBackendError as error:
            if error.status_code != 401 or not authorization.startswith("Bearer "):
                raise
        with self.session_lock:
            if not self.session:
                raise AttendanceBackendError("Management session expired", 401)
            if self._authorization() == authorization:
                try:
                    self.session = self._request(
                        "POST", "/auth/refresh", body={"refresh_token": self.session["refresh_token"]},
                    )
                except AttendanceBackendError as error:
                    if error.status_code == 401:
                        self.session = None
                    raise
            authorization = self._authorization()
        return self._request(method, path, authorization=authorization, body=body)

    def login(self, username, password):
        with self.session_lock:
            self.session = self._request("POST", "/auth/login", body={"username": username, "password": password})
            user = self.session["user"]
            if user["role"] != "admin":
                MongoDbConfig.COMPANY_ID = user["company_id"]
            return user

    def logout(self):
        try:
            if self.session:
                self.authenticated_request("POST", "/auth/logout")
        finally:
            with self.session_lock:
                self.session = None
                MongoDbConfig.COMPANY_ID = self.default_company

    @staticmethod
    def provision_camera(api_key):
        from src.attendance.device_credentials import store_device

        identity = mongo_db._request("GET", "/devices/self", authorization="Device " + api_key)
        store_device(api_key, identity)
        return identity

    def activate_camera(self):
        from src.attendance.camera_outbox import CameraOutbox
        from src.attendance.device_credentials import read_device, read_protected

        self.device = read_device()
        if not self.device:
            raise AttendanceBackendError("Provision a camera API key in AttendanceManager first", 401)
        MongoDbConfig.COMPANY_ID = self.device["company_id"]
        AuthServiceConfig.SYSTEM_ID = self.device["system_id"]
        cached = read_protected("camera-settings.dat")
        identity = cached if cached and cached["device_id"] == self.device["device_id"] else self.device
        self.camera_company_ids = tuple(identity.get("attendance_company_ids") or self.device.get("attendance_company_ids") or [MongoDbConfig.COMPANY_ID])
        self.outbox = CameraOutbox(self.device["device_id"], self._call, self._on_attendance_saved)

    def reload_camera_key(self):
        from src.attendance.device_credentials import read_device

        if self.device:
            saved = read_device()
            if saved and saved["device_id"] == self.device["device_id"]:
                self.device = saved

    def _call(self, operation, *args, **kwargs):
        if self.device:
            self.reload_camera_key()
        result = self.authenticated_request(
            "POST", f"/internal/attendance/{operation}", {"args": args, "kwargs": kwargs},
        )
        if self.device and "attendance_company_ids" in result:
            self.camera_company_ids = tuple(result["attendance_company_ids"])
        return result["result"]

    def get_settings(self, keys, username):
        settings = self._call("get_settings", keys, username)
        if self.device and tuple(keys) == CameraConfig.WATCHED_SETTING_KEYS:
            from src.attendance.device_credentials import write_protected

            try:
                write_protected("camera-settings.dat", {
                    "device_id": self.device["device_id"], "settings": settings,
                    "attendance_company_ids": self.camera_company_ids,
                })
            except OSError as error:
                logger.warning("Could not cache camera settings ({})", type(error).__name__)
        return settings

    def load_camera_config(self):
        from src.attendance.device_credentials import read_protected

        try:
            settings = self.get_settings(CameraConfig.WATCHED_SETTING_KEYS, MongoDbConfig.COMPANY_ID)
        except AttendanceBackendError as error:
            if error.status_code in {401, 403}:
                raise
            cached = read_protected("camera-settings.dat")
            settings = cached["settings"] if cached and cached["device_id"] == self.device["device_id"] else {}
        return CameraConfig.load_from_mongodb(self, settings=settings)

    def get_attendance_status(self, user_id, company_id=None):
        return self._call("get_attendance_status", user_id, company_id or MongoDbConfig.COMPANY_ID)

    def get_logs(self, company_id=None, date=None, start_date=None, end_date=None):
        return self._call("get_logs", company_id or MongoDbConfig.COMPANY_ID, date, start_date, end_date)

    def get_todays_logs(self, company_id=None):
        return self._call("get_todays_logs", company_id or MongoDbConfig.COMPANY_ID)

    def save_system_log(self, message, status_code, log_type="ERROR", priority="MEDIUM", devices_info=None):
        return self._call(
            "save_system_log", message, status_code, log_type, priority, devices_info,
            company_id=MongoDbConfig.COMPANY_ID, system_id=AuthServiceConfig.SYSTEM_ID,
        )

    def save_pending_license(self, **kwargs):
        kwargs["company_id"] = kwargs.get("company_id") or MongoDbConfig.COMPANY_ID
        kwargs["system_id"] = kwargs.get("system_id") or AuthServiceConfig.SYSTEM_ID
        return self._call("save_pending_license", **kwargs)

    def log_attendance(self, user_id, user_name, status=None, frame=None, company_id=None,
                       unknown_attempt=0, video_path=None, *, captured_at=None):
        image_blob = None
        if frame is not None:
            try:
                success, encoded = cv2.imencode(".webp", frame, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
                if success:
                    image_blob = encoded.tobytes()
            except cv2.error:
                logger.error("Failed to compress attendance image")

        arguments = (
            user_id, user_name, status, image_blob,
            company_id or MongoDbConfig.COMPANY_ID, unknown_attempt, video_path,
        )
        if self.outbox:
            self.outbox.put(arguments, {"event_id": uuid4().hex, "captured_at": captured_at or datetime.now(timezone.utc)})
            return status
        status, log_entry = self._call("log_attendance", *arguments)
        if log_entry:
            self._on_attendance_saved(log_entry, image_blob)
        return status

    def _on_attendance_saved(self, log_entry, image_blob):
        log_entry["image_webp"] = image_blob
        self._trigger_background_sync(log_entry)
        if self.attendance_notification:
            self.attendance_notification(log_entry, image_blob)

    def set_employee_active_status(self, user_id, company_id, active=True):
        success, message = self._call("set_employee_active_status", user_id, company_id, active)
        if success:
            try:
                from src.attendance.qdrant_db import attendance as qdrant_mgr

                qdrant_mgr.set_user_active_status(user_id, active)
            except Exception as error:
                return False, str(error)
        return success, message

    def _trigger_background_sync(self, log_entry):
        """
        Triggers a background thread to upload the log to HKB services.
        - Employees: Synchronized to their respective company's HKB services.
        - Strangers (Unknown): Saved locally without HKB upload.
        """
        import threading
        
        def sync_task():
            try:
                try: 
                    from src.services.hkb_service import hkb_service
                    from src.services.report_service import report_service
                except Exception as e: 
                    logger.error(f"Sync Task: Import error - {e}")
                    return
                
                user_id = log_entry.get("user_id")
                cid = log_entry.get("company_id")
                event_scope = {"company_id": cid} if self.device else {}
                attempt = log_entry.get("unknown_attempt", 0)

                # --- ALWAYS REPORT ATTENDANCE EVENT TO DASHBOARD (Even in TEST_MODE) ---
                image_url = log_entry.get("image_url")
                
                logger.info(f"Sync Task: Đang gọi report_info cho nhân viên {user_id} với image_url={image_url}")
                
                report_service.report_info(
                    message=f"Chấm công thành công: {log_entry['user_name']} (Mã NV: {user_id}) - Trạng thái: {log_entry['status']}",
                    status_code=200,
                    image_url=image_url
                )

                # TEST_MODE check (Only skip HKB Sync, keep Dashboard reporting)
                from src.config import RecognitionConfig
                if RecognitionConfig.TEST_MODE:
                    logger.debug("Sync: TEST_MODE is enabled. Skipping cloud HKB sync but reported to dashboard.")
                    return
                
                if user_id in ["Unknown", "Spoof"]:
                    # STRANGER/SPOOF SYNC POLICY: LOCAL ONLY (Do not sync to cloud)
                    logger.debug(f"Sync: Stranger/Spoof detected (Attempt {attempt}). Saving locally only, skipping cloud sync.")
                    return
                else:
                    # Known User Sync
                    user_companies = self.get_user_company_ids(user_id, **event_scope)
                    if cid and cid not in user_companies:
                        user_companies.append(str(cid))
                    services = self.get_auth_services_for_companies(user_companies)
                
                if not services:
                    logger.debug(f"Sync: No HKB services found for sync (User: {user_id}, Company: {cid})")
                    return

                for service in services:
                    logger.info(f"Sync: Auto-uploading attendance to {service['app_name']}...")
                    
                    # Gọi dịch vụ upload
                    res = hkb_service.upload_timekeepers(
                        endpoint=service["endpoint"],
                        system_id=service["uuid"],
                        api_key=service["key"],
                        user_id=service.get("user_id", 1),
                        attendance_logs=[log_entry]
                    )
                    
                    if res and res.success:
                        self.mark_logs_uploaded([log_entry["_id"]], service["uuid"], details={"status": "SUCCESS", "message": res.message}, **event_scope)
                        logger.success(f"Sync: Successfully synced to {service['app_name']}")

                        # TRÍCH XUẤT URL ẢNH ĐỂ GỬI LÊN DASHBOARD
                        image_url = None
                        if hasattr(res, 'data') and isinstance(res.data, dict):
                            image_url = res.data.get('image_url') or res.data.get('url') or res.data.get('link')
                        
                        if image_url:
                            report_service.report_info(
                                message=f"Đã có ảnh chấm công: {log_entry['user_name']} ({service['app_name']})",
                                status_code=200,
                                image_url=image_url
                            )
                    else:
                        msg = res.message if res else 'No response'
                        err_detail = None
                        if res:
                            try:
                                # Extract raw data or details from the auth SDK response, if present
                                err_detail = getattr(res, 'data', None) or getattr(res, 'raw', None)
                            except: pass
                            
                        details_obj = {"status": "FAILED", "message": msg}
                        if err_detail:
                            details_obj["error_detail"] = err_detail
                            
                        self.mark_logs_uploaded([log_entry["_id"]], service["uuid"], details=details_obj, **event_scope)
                        logger.warning(f"Sync: Failed to sync to {service['app_name']}: {msg}")
                        
                        # CHỈ GỬI LOG KHI ĐỒNG BỘ THẤT BẠI
                        report_service.report_error(
                            message=f"Lỗi đồng bộ chấm công: {log_entry['user_name']} -> {service['app_name']}. Lỗi: {msg}",
                            status_code=502
                        )
            
            except Exception as e:
                logger.error(f"Sync: Background task error: {e}")

        threading.Thread(target=sync_task, daemon=True).start()


mongo_db = AttendanceBackendClient()
