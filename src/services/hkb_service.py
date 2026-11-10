import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from loguru import logger
import requests

from bittech_auth import HKBClient
from src.config import AuthServiceConfig, MongoDbConfig
from src.attendance.mongodb_mgr import mongo_db


# =====================================================================
# Exceptions (Bittech Auth SDK Spec)
# =====================================================================

class BittechAuthError(Exception):
    code = "BITTECH_AUTH_ERROR"


class LicenseConfigError(BittechAuthError):
    code = "LICENSE_CONFIG_ERROR"


class LicenseNotFoundError(BittechAuthError):
    code = "LICENSE_NOT_FOUND"


class LicenseNotActiveError(BittechAuthError):
    code = "LICENSE_NOT_ACTIVE"


class ReplayAttackError(BittechAuthError):
    code = "REPLAY_ATTACK"


class AuthServiceRequestError(BittechAuthError):
    code = "AUTH_SERVICE_REQUEST_ERROR"


# =====================================================================
# Canonical JSON, License Seed & Security Helpers
# =====================================================================

def canonical_json_stringify(value: Any) -> str:
    """
    Chuẩn hóa JSON deterministic (sắp xếp key, không khoảng trắng dư thừa, UTF-8).
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256(value: str) -> str:
    """
    Tính SHA-256 hash của chuỗi ký tự UTF-8.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def calculate_license_seed(config: dict[str, Any]) -> str:
    """
    Tính license_seed = sha256(canonical_json_stringify(config)).
    """
    return sha256(canonical_json_stringify(config))


def load_license_config(path: str | Path) -> dict[str, Any]:
    """
    Đọc file cấu hình JSON của license.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise LicenseConfigError(f"License config file not found: {target_path}")
    
    try:
        with target_path.open("r", encoding="utf-8") as file:
            value = json.load(file)

        if not isinstance(value, dict):
            raise LicenseConfigError("License config must be a JSON object")

        return value
    except json.JSONDecodeError as e:
        raise LicenseConfigError(f"Invalid JSON in license config: {e}")


def load_license_seed(path: str | Path) -> Tuple[str, dict[str, Any]]:
    """
    Đọc license config và tính toán license_seed.
    """
    config = load_license_config(path)
    return calculate_license_seed(config), config


def create_request_metadata() -> dict[str, str]:
    """
    Tạo metadata chống replay attack:
    - nonce: 16 bytes ngẫu nhiên biểu diễn dạng hex (32 ký tự).
    - timestamp: Unix timestamp tính theo giây dạng string.
    """
    return {
        "nonce": secrets.token_hex(16),
        "timestamp": str(int(time.time())),
    }


# =====================================================================
# HKBService
# =====================================================================

class HKBService:
    """
    Adapter service for HKB Auth SDK and Bittech AuthService.
    Provides methods to manage licenses, authenticate, and sync attendance logs to cloud.
    """

    def __init__(self):
        self.auth_api_url = AuthServiceConfig.BASE_URL.rstrip("/")
        self.system_id = AuthServiceConfig.SYSTEM_ID
        self.client = HKBClient(base_url=self.auth_api_url)
        self.loop = asyncio.new_event_loop()

    def _run_sync(self, coro):
        """Helper to run async code in a synchronous manner."""
        try:
            if self.loop.is_closed():
                return None
            return self.loop.run_until_complete(coro)
        except Exception as e:
            if "after shutdown" not in str(e):
                logger.error(f"HKB Service: Sync execution error: {e}")
            return None

    # =================================================================
    # License Guard & License Management
    # =================================================================

    def require_active_license(self, system_id: Optional[str] = None) -> dict[str, Any]:
        """
        License Guard: Kiểm tra bản ghi license đang active.
        Ném ngoại lệ LicenseNotActiveError hoặc trả về dict thông tin license.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        license_record = mongo_db.find_active_license(now=now, system_id=system_id or self.system_id)

        if not license_record:
            # Thử tìm license gần nhất xem có bị hết hạn hay chưa kích hoạt
            latest = mongo_db.find_latest_license(system_id=system_id or self.system_id)
            if not latest:
                raise LicenseNotFoundError("Không tìm thấy thông tin License trong hệ thống.")
            if latest.get("status") != 1:
                raise LicenseNotActiveError("License chưa được kích hoạt (Status != 1).")
            expired_at = latest.get("expired_at")
            if expired_at and expired_at <= now:
                raise LicenseNotActiveError(f"License đã hết hạn vào {expired_at.strftime('%Y-%m-%d %H:%M:%S')}.")
            raise LicenseNotActiveError("License không hợp lệ hoặc chưa active.")

        return license_record

    def get_license_info(self, system_id: Optional[str] = None) -> dict[str, Any]:
        """
        Lấy tóm tắt trạng thái License hiện tại cho UI hiển thị.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        latest = mongo_db.find_latest_license(system_id=system_id or self.system_id)
        if not latest:
            return {
                "has_license": False,
                "status": -1,
                "status_label": "Chưa đăng ký",
                "is_active": False,
                "message": "Hệ thống chưa đăng ký License",
                "license_key": None,
                "expired_at": None,
                "days_left": 0
            }

        status = latest.get("status", 0)
        expired_at = latest.get("expired_at")
        days_left = (expired_at - now).days if (expired_at and isinstance(expired_at, datetime)) else 0
        is_expired = expired_at is not None and expired_at <= now

        if status == 1 and not is_expired:
            status_label = "Đang hoạt động (Active)"
            is_active = True
            msg = f"License hợp lệ. Còn lại {days_left} ngày."
        elif is_expired:
            status_label = "Đã hết hạn (Expired)"
            is_active = False
            msg = f"License đã hết hạn vào {expired_at.strftime('%d/%m/%Y')}."
        elif status == 0:
            status_label = "Chờ kích hoạt (Pending)"
            is_active = False
            msg = "License đã đăng ký nhưng chưa được xác thực."
        else:
            status_label = "Đã vô hiệu hóa (Revoked)"
            is_active = False
            msg = "License đã bị thu hồi."

        return {
            "has_license": True,
            "status": status,
            "status_label": status_label,
            "is_active": is_active,
            "message": msg,
            "license_key": latest.get("license_key"),
            "public_id": latest.get("public_id"),
            "label": latest.get("label"),
            "expired_at": expired_at.strftime("%Y-%m-%d %H:%M:%S") if expired_at else None,
            "verified_at": latest.get("verified_at").strftime("%Y-%m-%d %H:%M:%S") if latest.get("verified_at") else None,
            "days_left": max(0, days_left)
        }

    def register_license(
        self,
        label: str = "Face Attendance System",
        expired_at: Optional[str | datetime] = None,
        notes: str = "",
        config_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        Đăng ký License mới với AuthService theo đặc tả Bittech Auth SDK.
        1. Đọc file config và tính license_seed.
        2. Gửi multipart/form-data lên /api/license/register.
        3. Lưu bản ghi vào MongoDB với status = 0 (Pending).
        4. Tự động gọi verify_license để kích hoạt ngay.
        """
        try:
            cfg_path = Path(config_path or AuthServiceConfig.LICENSE_CONFIG_PATH)
            if not cfg_path.exists():
                raise LicenseConfigError(f"Không tìm thấy file cấu hình license tại: {cfg_path}")

            seed, _ = load_license_seed(cfg_path)
            
            # Format expired_at string
            if expired_at is None:
                # Mặc định 1 năm nếu không truyền
                from datetime import timedelta
                exp_dt = datetime.utcnow() + timedelta(days=365)
                exp_str = exp_dt.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(expired_at, datetime):
                exp_dt = expired_at
                exp_str = exp_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                exp_str = str(expired_at)
                try:
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    exp_dt = datetime.utcnow()

            register_url = f"{self.auth_api_url}/api/license/register"
            logger.info(f"HKB Service: Registering license at {register_url} with seed {seed[:10]}...")

            with cfg_path.open("rb") as f:
                files = {
                    "license_config": (
                        cfg_path.name,
                        f,
                        "application/octet-stream",
                    )
                }
                data = {
                    "license_seed": seed,
                    "label": label,
                    "expired_at": exp_str,
                    "notes": notes,
                }
                headers = {"Accept": "application/json"}
                response = requests.post(register_url, headers=headers, files=files, data=data, timeout=15)

            if response.status_code not in (200, 201):
                logger.error(f"HKB Service: Register license HTTP {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "message": f"Đăng ký thất bại (HTTP {response.status_code}): {response.text}",
                    "data": None
                }

            res_json = response.json()
            payload_data = res_json.get("data", res_json)
            license_key = payload_data.get("license_key") or payload_data.get("key")
            public_id = payload_data.get("public_id") or payload_data.get("id") or secrets.token_hex(8)
            signature = payload_data.get("signature")

            if not license_key:
                logger.error(f"HKB Service: Register license response missing license_key. Response: {res_json}")
                return {
                    "success": False,
                    "message": "Phản hồi từ server thiếu license_key.",
                    "data": payload_data
                }

            # Lưu vào MongoDB với trạng thái PENDING (0)
            mongo_db.save_pending_license(
                public_id=str(public_id),
                license_key=str(license_key),
                license_seed=str(seed),
                label=label,
                expired_at=exp_dt,
                signature=signature,
                notes=notes,
                system_id=self.system_id,
                company_id=MongoDbConfig.COMPANY_ID,
            )
            logger.success(f"HKB Service: Registered license {license_key}. Now verifying...")

            # Tự động kích hoạt Verify License
            verify_res = self.verify_license(license_key=license_key, signature=signature, config_path=cfg_path)
            return {
                "success": True,
                "message": "Đăng ký và kích hoạt license thành công." if verify_res.get("success") else "Đăng ký thành công, verify đang chờ.",
                "data": payload_data,
                "verify_result": verify_res
            }

        except Exception as e:
            logger.error(f"HKB Service: Register license exception: {e}")
            return {
                "success": False,
                "message": f"Lỗi khi đăng ký license: {str(e)}",
                "data": None
            }

    def verify_license(
        self,
        license_key: Optional[str] = None,
        signature: Optional[str] = None,
        config_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        Xác thực và kích hoạt License với AuthService (/api/license/verify).
        """
        try:
            cfg_path = Path(config_path or AuthServiceConfig.LICENSE_CONFIG_PATH)
            seed = ""
            if cfg_path.exists():
                seed, _ = load_license_seed(cfg_path)

            # Nếu không truyền license_key, tìm trong DB bản ghi mới nhất
            if not license_key:
                latest = mongo_db.find_latest_license(system_id=self.system_id)
                if not latest:
                    raise LicenseNotFoundError("Không tìm thấy license trong DB để verify.")
                license_key = latest.get("license_key")
                signature = signature or latest.get("signature")
                if not seed:
                    seed = latest.get("license_seed", "")

            metadata = create_request_metadata()
            verify_url = f"{self.auth_api_url}/api/license/verify"
            payload = {
                "license_key": license_key,
                "license_seed": seed,
                "system_id": self.system_id,
                "nonce": metadata["nonce"],
                "timestamp": metadata["timestamp"],
                "signature": signature or "",
            }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            logger.info(f"HKB Service: Verifying license {license_key} at {verify_url}...")
            response = requests.post(verify_url, headers=headers, json=payload, timeout=15)

            if response.status_code in (200, 201):
                res_json = response.json()
                verified_at = datetime.utcnow()
                mongo_db.mark_license_verified(license_key, verified_at=verified_at, signature=signature)
                logger.success(f"HKB Service: License {license_key} verified successfully.")
                return {
                    "success": True,
                    "message": "Xác thực license thành công.",
                    "data": res_json.get("data", res_json)
                }
            else:
                logger.warning(f"HKB Service: Verify license failed (HTTP {response.status_code}): {response.text}")
                return {
                    "success": False,
                    "message": f"Xác thực thất bại (HTTP {response.status_code}): {response.text}",
                    "data": None
                }

        except Exception as e:
            logger.error(f"HKB Service: Verify license exception: {e}")
            return {
                "success": False,
                "message": f"Lỗi xác thực license: {str(e)}",
                "data": None
            }

    # =================================================================
    # Client & System Connection APIs
    # =================================================================

    def register_client(self, system_id, external_id, description, user_info, system_connection_id=1, system_register="face_recognition"):
        """
        Register a new client system with replay metadata and license key.
        """
        try:
            # Lấy thông tin active license nếu có
            license_rec = None
            try:
                license_rec = self.require_active_license()
            except Exception as le:
                logger.warning(f"HKB Service: Registering client without active license: {le}")

            logger.info(f"HKB Service: Registering client. cid={system_connection_id}, sys_id={system_id}, ext_id={external_id}")
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
                logger.info(f"HKB Service Response: Success={result.success}, Message={result.message}")
                
                # Save to MongoDB if successful
                if result.success and result.data:
                    try:
                        data = result.data
                        sys_conn = data.get("system_connection", {})
                        
                        auth_data = {
                            "uuid": sys_conn.get("system_id", system_id),
                            "endpoint": sys_conn.get("endpoint"),
                            "connection_type": "hkb",
                            "key": data.get("api_key"),
                            "user_id": data.get("external_id"),
                            "app_name": sys_conn.get("name"),
                            "app_info": json.dumps(data, ensure_ascii=False)
                        }
                        mongo_db.save_auth_service(auth_data)
                        
                        system_name = auth_data.get("app_name")
                        remote_system_id = auth_data.get("uuid")
                        
                        if system_name and remote_system_id:
                            existing_company = mongo_db.get_company(remote_system_id)
                            if not existing_company:
                                logger.info(f"HKB Service: Creating local company for system {system_name} ({remote_system_id})")
                                mongo_db.create_company(
                                    company_id=remote_system_id, 
                                    name=system_name, 
                                    description=f"Connected from HKB Service: {system_name}"
                                )

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
            return result
        except Exception as e:
            logger.error(f"HKB Service: Register client failed: {e}")
            return None

    def authenticate(self, system_id, api_key, user_id):
        """
        Authenticate a user / system against Central AuthService.
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
                mongo_db.delete_auth_service(system_id)
            return result
        except Exception as e:
            logger.error(f"HKB Service: Revoke connection failed: {e}")
            return None

    def get_connections(self, user_id: int = 1, username: str = "GLOBAL"):
        """
        Get list of connections for the current API Key.
        """
        try:
            custom_keys = mongo_db.get_setting("group_keys", "", username=MongoDbConfig.COMPANY_ID)
            if not custom_keys and username and username != MongoDbConfig.COMPANY_ID:
                custom_keys = mongo_db.get_setting("group_keys", "", username=username)

            if custom_keys:
                group_keys = [k.strip() for k in custom_keys.split(",") if k.strip()]
            else:
                group_keys = [AuthServiceConfig.API_KEY]

            auth_records = mongo_db.get_auth_services_for_user(user_id)
            client_registers = [r["uuid"] for r in auth_records]
            
            if not client_registers:
                client_registers = [AuthServiceConfig.SYSTEM_ID]

            logger.info(f"HKB Service: Fetching connections list. group_keys: {group_keys}, user_id: {user_id}")
            coro = self.client.get_system_connections(
                group_key=group_keys,
                client_registers=client_registers,
            )
            result = self._run_sync(coro)
            if result and result.success:
                return result.data
            return None
        except Exception as e:
            logger.error(f"HKB Service: Get connections failed: {e}")
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
            return self._run_sync(coro)
        except Exception as e:
            logger.error(f"HKB Service: Fetch employees failed: {e}")
            return None

    def get_hr_users(self, endpoint, system_id, api_key, user_id):
        """
        Fetch HR user list from external endpoint.
        """
        try:
            logger.info(f"HKB Service: Fetching HR users from {endpoint}")
            coro = self.client.get_hr_users(
                endpoint=endpoint,
                system_id=system_id,
                api_key=api_key,
                user_id=user_id
            )
            return self._run_sync(coro)
        except Exception as e:
            logger.error(f"HKB Service: Get HR users failed: {e}")
            return None

    # =================================================================
    # Attendance Cloud Sync
    # =================================================================

    def upload_timekeepers(self, endpoint, system_id, api_key, user_id, attendance_logs):
        """
        Upload attendance logs (timekeepers) to external system with WebP images.
        Áp dụng License Guard và cơ chế auto re-authentication khi token hết hạn.
        """
        try:
            # 1. License Guard Check
            try:
                self.require_active_license()
            except LicenseNotActiveError as le:
                logger.error(f"HKB Service: Chặn upload chấm công do License không hợp lệ: {le}")
                return None
            except Exception as le:
                logger.warning(f"HKB Service: License guard check warning: {le}")

            url = f"{endpoint.rstrip('/')}/api/hr/timekeepers"
            logger.info(f"HKB Service: Uploading {len(attendance_logs)} timekeepers to {url}")
            
            # 2. Prepare payload data
            payload_data = []
            files_data = {}
            
            for log in attendance_logs:
                io_status = log.get("status", "IN")
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
                
                image_blob = log.get("image_webp")
                if image_blob:
                    session_id = log.get("session_id")
                    files_data[session_id] = image_blob
            
            coro = self.client.upload_timekeepers(
                url=url,
                system_id=system_id,
                api_key=api_key,
                user_id=user_id,
                payload=payload_data,
                files=files_data
            )
            result = self._run_sync(coro)
            
            # 3. Auto Re-Authentication Logic if Token Expired
            token_error_keywords = ["token", "expired", "unauthorized", "401", "403", "truy cập bị từ chối", "hết hạn", "xác thực"]
            is_token_error = False
            if result and not result.success and result.message:
                msg_lower = result.message.lower()
                if any(k in msg_lower for k in token_error_keywords):
                    is_token_error = True
            
            if not result or is_token_error:
                logger.warning(f"HKB Service: Upload failed (token error: {is_token_error}). Re-authenticating for {system_id}...")
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

            if result:
                logger.info(f"HKB Service: Upload result: Success={result.success}, Message={result.message}")
                if not result.success:
                    from src.utils.telegram_bot import send_telegram_report
                    for log in attendance_logs:
                        send_telegram_report(
                            "Sync Fail", 
                            f"Đồng bộ lên server thất bại!\nNhân viên: {log.get('user_name')} ({log.get('user_id')})\nStatus: {log.get('status')}\nServer Message: {result.message}",
                            image=log.get("image_webp")
                        )
            else:
                logger.warning("HKB Service: Upload timekeepers returned None after retry")
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

    def sync_pending_logs(self, system_id: str, endpoint: str, api_key: str, user_id: int = 1, batch_size: int = 50) -> dict[str, Any]:
        """
        Đồng bộ bù các log chấm công tồn đọng (offline / pending) lên Cloud.
        """
        try:
            pending_logs = mongo_db.get_unuploaded_logs(system_id=system_id, limit=batch_size)
            if not pending_logs:
                return {"success": True, "synced_count": 0, "message": "Không có log tồn đọng cần sync."}

            logger.info(f"HKB Service: Bắt đầu sync {len(pending_logs)} log tồn đọng cho system {system_id}...")
            res = self.upload_timekeepers(
                endpoint=endpoint,
                system_id=system_id,
                api_key=api_key,
                user_id=user_id,
                attendance_logs=pending_logs
            )

            if res and res.success:
                log_ids = [str(l["_id"]) for l in pending_logs if "_id" in l]
                mongo_db.mark_logs_uploaded(log_ids, system_id, details={"status": "SUCCESS", "message": res.message})
                return {"success": True, "synced_count": len(pending_logs), "message": f"Đã sync thành công {len(pending_logs)} logs."}
            else:
                msg = res.message if res else "Không có phản hồi từ server"
                return {"success": False, "synced_count": 0, "message": f"Sync thất bại: {msg}"}
        except Exception as e:
            logger.error(f"HKB Service: sync_pending_logs failed: {e}")
            return {"success": False, "synced_count": 0, "message": str(e)}


# Global instance
hkb_service = HKBService()
