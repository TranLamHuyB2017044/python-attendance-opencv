import asyncio
from bittech_auth import HKBClient
from src.config import AuthServiceConfig
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
            return self.loop.run_until_complete(coro)
        except Exception as e:
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

    def get_connections(self, user_id: int = 1):
        """
        Get list of connections for the current API Key.
        """
        try:
            # Lấy danh sách Group Keys từ settings (nếu có)
            custom_keys = mongo_db.get_setting("group_keys", "")
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

# Global instance
hkb_service = HKBService()
