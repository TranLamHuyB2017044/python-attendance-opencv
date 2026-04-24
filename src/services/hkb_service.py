import asyncio
from bittech_auth import HKBClient
from src.config import AuthServiceConfig, MongoDbConfig
from loguru import logger
import os
import json
from src.attendance.mongodb_mgr import mongo_db

class HKBService:
    """
    Adapter service for HKB Auth SDK.
    Provides synchronous methods to interact with the HKB Auth system.
    """
    def __init__(self):
        self.client = HKBClient(
            base_url=AuthServiceConfig.BASE_URL
        )
        self.loop = asyncio.new_event_loop()

    def _run_sync(self, coro):
        """Helper to run async code in a synchronous manner."""
        try:
            if self.loop.is_closed():
                return None
            return self.loop.run_until_complete(coro)
        except Exception as e:
            # Silent failure during shutdown
            if "after shutdown" not in str(e):
                logger.error(f"HKB Service: Sync execution error: {e}")
            return None

    def register_client(self, system_id, external_id, description, user_info, system_connection_id=1, system_register="face_recognition"):
        """
        Register a new client system.
        """
        try:
            logger.info(f"HKB Service: Registering client. Data: cid={system_connection_id}, sys_id={system_id}, reg={system_register}, ext_id={external_id}, desc={description}, info={user_info}")
            coro = self.client.register_client(
                system_connection_id=system_connection_id,
                system_id=system_id,
                system_register=system_register,
                external_id=external_id,
                description=description,
                user_info=user_info
            )
            result = self._run_sync(coro)
            if result:
                logger.info(f"HKB Service Response: Success={result.success}, Message={result.message}, Data={result.data}")
                
                # Save to MongoDB if successful
                if result.success and result.data:
                    try:
                        data = result.data
                        sys_conn = data.get("system_connection", {})
                        
                        auth_data = {
                            "uuid": sys_conn.get("system_id", system_id), # uuid is systemId
                            "endpoint": sys_conn.get("endpoint"),         # endpoint is endpoint
                            "connection_type": "hkb",
                            "key": data.get("api_key"),                   # api key lưu vào key
                            "user_id": data.get("external_id"),
                            "app_name": sys_conn.get("name"),             # appname là name
                            "app_info": json.dumps(data, ensure_ascii=False) # app info là full response kiểu json
                        }
                        mongo_db.save_auth_service(auth_data)
                        
                        # Create company based on system connection ONLY if:
                        # 1. We are using the admin token (AuthServiceConfig.API_KEY is present)
                        # 2. The company doesn't exist yet
                        system_name = auth_data.get("app_name")
                        remote_system_id = auth_data.get("uuid")
                        
                        if system_name and remote_system_id:
                            # Check if company exists
                            existing_company = mongo_db.companies.find_one({"company_id": remote_system_id})
                            
                            # Allow company creation if it doesn't exist.
                            # The UI already restricts who can reach this registration step.
                            if not existing_company:
                                logger.info(f"HKB Service: Creating local company for system {system_name} ({remote_system_id})")
                                mongo_db.create_company(
                                    company_id=remote_system_id, 
                                    name=system_name, 
                                    description=f"Connected from HKB Service: {system_name}"
                                )
                            else:
                                logger.info(f"HKB Service: Company {remote_system_id} already exists, skipping creation.")

                        # Tự động gọi authenticate để lấy token sau khi kết nối thành công
                        remote_system_id = auth_data["uuid"]
                        remote_api_key = auth_data["key"]
                        logger.info(f"HKB Service: Automatically authenticating for system {remote_system_id}")
                        self.authenticate(
                            system_id=remote_system_id,
                            api_key=remote_api_key,
                            user_id=data.get("external_id", external_id)
                        )
                    except Exception as e:
                        logger.error(f"HKB Service: Failed to process/save response data or authenticate: {e}")
            else:
                logger.warning("HKB Service Response: None (Sync execution failed)")
            return result
        except Exception as e:
            logger.error(f"HKB Service: Register client failed: {e}")
            return None

    def authenticate(self, system_id, api_key, user_id):
        """
        Authenticate a user.
        """
        try:
            logger.info(f"HKB Service: Authenticating user {user_id} for system {system_id}")
            coro = self.client.authenticate(
                system_id=system_id,
                api_key=api_key,
                user_id=user_id
            )
            auth = self._run_sync(coro)
            if auth:
                status_icon = "SUCCESS" if auth.success else "FAILED"
                logger.info(f"HKB Service: Authenticate result ({status_icon}): {auth.message}")
                if auth.data:
                    logger.debug(f"HKB Service: Authenticate Response Data: {json.dumps(auth.data, ensure_ascii=False)}")
                if not auth.success and hasattr(auth, 'status_code'):
                    logger.warning(f"HKB Service: Authenticate failed with Status Code: {auth.status_code}")
            return auth
        except Exception as e:
            logger.error(f"HKB Service: Authenticate failed: {e}")
            return None

    def revoke_connection(self, system_id, api_key, password):
        """
        Revoke an existing connection/API key.
        """
        try:
            logger.info(f"HKB Service: Revoking connection for system {system_id}")
            coro = self.client.revoke_api_key(
                system_id=system_id,
                api_key=api_key,
                password=password
            )
            result = self._run_sync(coro)
            
            if result and result.success:
                logger.success(f"HKB Service: Successfully revoked connection for {system_id}")
                # Optional: remove from local DB
                mongo_db.auth_services.delete_one({"uuid": system_id})
            
            return result
        except Exception as e:
            logger.error(f"HKB Service: Revoke connection failed: {e}")
            return None

    def get_connections(self, user_id: int = 1, username: str = "GLOBAL"):
        """
        Get list of connections for the current API Key.
        """
        try:
            # Lấy danh sách Group Keys từ settings
            # Ưu tiên lấy theo COMPANY_ID (Machine Scoped) để khớp với UI Settings
            custom_keys = mongo_db.get_setting("group_keys", "", username=MongoDbConfig.COMPANY_ID)
            
            # Nếu không có và username khác COMPANY_ID, thử tìm theo username (fallback legacy)
            if not custom_keys and username and username != MongoDbConfig.COMPANY_ID:
                custom_keys = mongo_db.get_setting("group_keys", "", username=username)

            if custom_keys:
                # Tách chuỗi comma-separated thành list và loại bỏ khoảng trắng
                group_keys = [k.strip() for k in custom_keys.split(",") if k.strip()]
            else:
                group_keys = [AuthServiceConfig.API_KEY]

            # Lấy danh sách UUID đã đăng ký từ DB theo user_id
            auth_records = list(mongo_db.auth_services.find({"user_id": user_id}))
            client_registers = [r["uuid"] for r in auth_records]
            
            # Nếu chưa có bản ghi nào, mặc định dùng SYSTEM_ID để có thể thấy danh sách chung
            if not client_registers:
                client_registers = [AuthServiceConfig.SYSTEM_ID]

            logger.info(f"HKB Service: Fetching connections list. group_keys: {group_keys}, user_id: {user_id}")
            coro = self.client.get_system_connections(
                group_key=group_keys, # Giờ truyền list string
                client_registers=client_registers,
            )
            result = self._run_sync(coro)
            
            if result:
                logger.info(f"HKB Service Response: Success={result.success}, Message={result.message}, Data={result.data}")
            else:
                logger.warning("HKB Service Response: None (Sync execution failed or returned invalid data)")

            if result and result.success:
                return result.data
            return None
        except Exception as e:
            logger.error(f"HKB Service: Get connections failed: {e}")
            return None

    def upload_attendance(self, user_id, user_name, timestamp, status, image_path=None):
        """
        Upload attendance log to HKB service by sharing it to an external endpoint.
        """
        try:
            logger.info(f"HKB Service: Uploading attendance for {user_name} ({user_id})")
            
            # Since the current SDK uses upload_document and expects an endpoint,
            # we might need to find which connection to use.
            connections = self.get_connections()
            if not connections or not isinstance(connections, list) or len(connections) == 0:
                logger.warning("HKB Service: No connections found to upload to.")
                return None
            
            # Use the first connection for demonstration
            conn = connections[0]
            endpoint = conn.get("endpoint")
            if not endpoint:
                logger.warning("HKB Service: Connection has no endpoint.")
                return None

            payload = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp,
                "status": status,
                "app_name": "Face Attendance System"
            }
            
            # If there's an image, we might need a different method or encode it
            # The current upload_document in SDK only takes a dict payload.
            # Usually images are uploaded as Multipart, but let's stick to what's in SDK.
            
            coro = self.client.upload_document(
                endpoint=endpoint,
                system_id=AuthServiceConfig.SYSTEM_ID,
                api_key=AuthServiceConfig.API_KEY,
                user_id=1, # Default user_id for system auth
                payload=payload
            )
            result = self._run_sync(coro)
            return result
        except Exception as e:
            logger.error(f"HKB Service: Upload attendance failed: {e}")
            return None

    def get_employees(self, endpoint, system_id, api_key, user_id):
        """
        Fetch employee list from external endpoint.
        """
        try:
            url = f"{endpoint.rstrip('/')}/api/hr/users"

            logger.info(f"HKB Service: Fetching employees from {url}")
            coro = self.client.get_employees(
                url=url,
                system_id=system_id,
                api_key=api_key,
                user_id=user_id
            )
            result = self._run_sync(coro)
            if result:
                logger.info(f"HKB Service: Get employees result: Success={result.success}")
                if result.data:
                    try:
                        logger.info(f"HKB Service: Employee List JSON: {json.dumps(result.data, ensure_ascii=False)}")
                    except Exception as e:
                        logger.error(f"Failed to log employee list JSON: {e}")
                
                if not result.success and result.data and "raw" in result.data:
                    logger.warning(f"HKB Service: Raw response: {result.data['raw'][:1000]}")
            return result
        except Exception as e:
            logger.error(f"HKB Service: Fetch employees failed: {e}")
            return None

    def get_hr_users(self, endpoint, system_id, api_key, user_id):
        """
        Fetch HR user list from external endpoint. 
        Intended for tester environment.
        """
        try:
            logger.info(f"HKB Service: Fetching HR users from {endpoint}")
            coro = self.client.get_hr_users(
                endpoint=endpoint,
                system_id=system_id,
                api_key=api_key,
                user_id=user_id
            )
            result = self._run_sync(coro)
            return result
        except Exception as e:
            logger.error(f"HKB Service: Get HR users failed: {e}")
            return None

    def upload_timekeepers(self, endpoint, system_id, api_key, user_id, attendance_logs):
        """
        Upload attendance logs (timekeepers) to external system with images.
        
        Args:
            endpoint: Base URL of the target system
            system_id: System ID for authentication
            api_key: API key for authentication
            user_id: User ID for authentication
            attendance_logs: List of attendance log dicts with session_id, user_id, user_name, timestamp, status, image_webp
        
        Returns:
            Result object with success status
        """
        try:
            url = f"{endpoint.rstrip('/')}/api/hr/timekeepers"
            logger.info(f"HKB Service: Uploading {len(attendance_logs)} timekeepers to {url}")
            
            # Prepare payload data
            payload_data = []
            files_data = {}
            
            for log in attendance_logs:
                # Map status to io field
                io_status = log.get("status", "IN")  # IN, OUT, or FAILED

                # Normalize datetime → server yêu cầu format "Y-m-d H:i:s" (không có milliseconds)
                # VD: "2026-04-24 11:31:03.361" → "2026-04-24 11:31:03"
                raw_ts = log.get("timestamp", "")
                try:
                    normalized_dt = str(raw_ts).split(".")[0]
                except Exception:
                    normalized_dt = str(raw_ts)

                payload_item = {
                    "session_id": log.get("session_id"),
                    "employee_code": log.get("user_id"),
                    "full_name": log.get("user_name"),
                    "datetime": normalized_dt,
                    "io": io_status
                }
                payload_data.append(payload_item)
                
                # Prepare image file if available
                image_blob = log.get("image_webp")
                if image_blob:
                    session_id = log.get("session_id")
                    files_data[session_id] = image_blob
            
            # Call SDK method
            logger.debug(f"HKB Service Request - SystemID: {system_id}, UserID: {user_id}")
            logger.debug(f"HKB Service Payload (first 2): {payload_data[:2]}")
            logger.debug(f"HKB Service Files count: {len(files_data)}")
            
            coro = self.client.upload_timekeepers(
                url=url,
                system_id=system_id,
                api_key=api_key,
                user_id=user_id,
                payload=payload_data,
                files=files_data
            )
            result = self._run_sync(coro)
            
            # --- AUTO RE-AUTHENTICATION LOGIC ---
            # If result is None or failed with a token error, try to re-authenticate and retry once
            token_error_keywords = ["token", "expired", "unauthorized", "401", "403", "truy cập bị từ chối", "hết hạn", "xác thực"]
            is_token_error = False
            if result and not result.success and result.message:
                msg_lower = result.message.lower()
                if any(k in msg_lower for k in token_error_keywords):
                    is_token_error = True
            
            if not result or is_token_error:
                logger.warning(f"HKB Service: Upload failed (token error: {is_token_error}). Attempting auto-reauth for system {system_id}...")
                auth_res = self.authenticate(system_id=system_id, api_key=api_key, user_id=user_id)
                
                if auth_res and auth_res.success:
                    logger.success(f"HKB Service: Re-auth successful, retrying upload...")
                    coro_retry = self.client.upload_timekeepers(
                        url=url,
                        system_id=system_id,
                        api_key=api_key,
                        user_id=user_id,
                        payload=payload_data,
                        files=files_data
                    )
                    result = self._run_sync(coro_retry)
                else:
                    logger.error(f"HKB Service: Auto-reauth failed for {system_id}")
            # ------------------------------------

            if result:
                logger.info(f"HKB Service: Upload result: Success={result.success}, Message={result.message}")
                if not result.success:
                    # Log everything inside the result object to see what's happening
                    try:
                        res_details = {attr: getattr(result, attr) for attr in dir(result) if not attr.startswith('__')}
                        logger.warning(f"HKB Service: Upload Full Result Object: {res_details}")
                    except Exception as le:
                        logger.error(f"Could not log full result: {le}")

                    # Send Telegram reports for each failed log entry
                    from src.utils.telegram_bot import send_telegram_report
                    for log in attendance_logs:
                        send_telegram_report(
                            "Sync Fail", 
                            f"Đồng bộ lên server thất bại!\nNhân viên: {log.get('user_name')} ({log.get('user_id')})\nStatus: {log.get('status')}\nServer Message: {result.message}",
                            image=log.get("image_webp")
                        )
            else:
                logger.warning("HKB Service: Upload timekeepers returned None after retry")
                # Send summary Telegram report for critical failure
                from src.utils.telegram_bot import send_telegram_report
                send_telegram_report(
                    "Sync Fail", 
                    f"Đồng bộ lên server thất bại hoàn toàn (Không có phản hồi từ API).\nSố lượng log: {len(attendance_logs)}"
                )
            
            return result
        except Exception as e:
            logger.error(f"HKB Service: Upload timekeepers failed: {e}")
            from src.utils.telegram_bot import send_telegram_report
            send_telegram_report("Sync Fail", f"Lỗi hệ thống khi đồng bộ: {str(e)}")
            return None

# Global instance
hkb_service = HKBService()
