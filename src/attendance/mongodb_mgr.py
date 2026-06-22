import pymongo
import datetime
import bcrypt
import pytz
from typing import Optional
from loguru import logger
import cv2
import numpy as np
from src.config import MongoDbConfig, RecognitionConfig, AuthServiceConfig

class MongoDBManager:
    """
    Manages attendance logs and user permissions using Cloud MongoDB.
    """
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MongoDbConfig.CONNECTION_STRING)
            self.db = self.client[MongoDbConfig.DATABASE_NAME]
            
            # Collections
            self.logs = self.db["attendance_logs"]
            self.companies = self.db["companies"]
            self.users = self.db["users"] # Admins and Company accounts
            self.auth_services = self.db["auth_services"] # HKB/Auth Connections
            self.settings = self.db["settings"] # System Settings
            self.employees = self.db["employees"] # Metadata for registered employees
            self.enrollment_images = self.db["enrollment_images"] # Enrollment images
            self.system_logs = self.db["system_logs"] # Logs for external monitoring
            
            # Create indexes for faster queries
            self.logs.create_index([("company_id", 1), ("date", -1)])
            self.logs.create_index([("user_id", 1)])
            self.users.create_index([("username", 1)], unique=True)
            self.auth_services.create_index([("uuid", 1)], unique=True)
            
            # Remove old unique index on key if exists, and create compound index (key, username)
            try:
                self.settings.drop_index("key_1")
            except:
                pass
            self.settings.create_index([("key", 1), ("username", 1)], unique=True)
            
            self.employees.create_index([("company_id", 1), ("user_id", 1)], unique=True)
            self.enrollment_images.create_index([("user_id", 1), ("company_id", 1)])
            
            logger.info("Connected to MongoDB Cloud successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")

    # --- Settings Methods ---
    def get_setting(self, key, default=None, username="GLOBAL"):
        """Retrieve a system setting, optionally user-specific."""
        try:
            # 1. Try to find user-specific setting
            setting = self.settings.find_one({"key": key, "username": username})
            if setting:
                return setting.get("value", default)
            
            # 2. Fallback to GLOBAL if not found and we were looking for a specific user
            if username != "GLOBAL":
                setting = self.settings.find_one({"key": key, "username": "GLOBAL"})
                if setting:
                    return setting.get("value", default)
                    
            return default
        except Exception as e:
            logger.error(f"Failed to get setting {key} for user {username}: {e}")
            return default

    def set_setting(self, key, value, username="GLOBAL"):
        """Save/Update a system setting, optionally user-specific."""
        try:
            self.settings.update_one(
                {"key": key, "username": username},
                {"$set": {"value": value, "updated_at": datetime.datetime.utcnow()}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set setting {key} for user {username}: {e}")
            return False

    def get_current_shift(self):
        """Returns the current shift name based on Vietnam time (Morning, Afternoon, Other)."""
        try:
            from src.utils.time_manager import time_mgr
            vn_now = time_mgr.get_accurate_time()
            hour_val = vn_now.hour * 100 + vn_now.minute
            
            # Ca sáng: 00:05 - 12:59
            if 5 <= hour_val <= 1259:
                return "Morning"
            # Ca chiều: 13:00 - 23:59
            elif 1300 <= hour_val <= 2359:
                return "Afternoon"
            else:
                return "Other"
        except Exception:
            return "Other"

    def get_attendance_status(self, user_id, company_id=None):
        """Checks the last log to determine if user should be IN, OUT or COOLDOWN with Shift Support."""
        try:
            from src.utils.time_manager import time_mgr
            vn_now = time_mgr.get_accurate_time()
            date_str = vn_now.strftime("%Y-%m-%d")
            cid = company_id or MongoDbConfig.COMPANY_ID
            
            # Current time in VN (HHMM) for shift logic
            current_time_val = vn_now.hour * 100 + vn_now.minute
            
            # 1. Fetch last record today
            last_record = self.logs.find_one(
                {"user_id": user_id, "company_id": cid, "date": date_str},
                sort=[("_id", -1)]
            )
            
            # 2. Check Cooldown
            if last_record:
                last_time = last_record.get("created_at")
                if last_time:
                    if isinstance(last_time, str):
                        try: last_time = datetime.datetime.fromisoformat(last_time)
                        except: last_time = None
                    
                    if last_time:
                        elapsed = (datetime.datetime.utcnow() - last_time).total_seconds()
                        if elapsed < RecognitionConfig.COOLDOWN_SECONDS:
                            return 'COOLDOWN'

            # 3. Shift Logic
            # Ca sáng: 00:05 - 12:59
            if 5 <= current_time_val <= 1259:
                morning_in = self.logs.find_one({
                    "user_id": user_id, 
                    "company_id": cid, 
                    "date": date_str,
                    "status": "IN"
                })
                if not morning_in:
                    return 'IN'
                
            # Ca chiều: 13:00 - 23:59
            elif 1300 <= current_time_val <= 2359:
                # Define start of afternoon shift in UTC
                vn_afternoon_start = vn_now.replace(hour=13, minute=0, second=0, microsecond=0)
                utc_afternoon_start = vn_afternoon_start.astimezone(pytz.utc).replace(tzinfo=None)
                
                afternoon_in = self.logs.find_one({
                    "user_id": user_id, 
                    "company_id": cid, 
                    "date": date_str,
                    "status": "IN",
                    "created_at": {"$gte": utc_afternoon_start}
                })
                if not afternoon_in:
                    return 'IN'

            # 4. Default: Toggle IN/OUT based on last record
            if last_record and last_record.get('status') == 'IN':
                return 'OUT'
            
            return 'IN'
        except Exception as e:
            logger.error(f"MongoDB: Error getting status: {e}")
            return 'IN'

    def log_attendance(self, user_id, user_name, status=None, frame=None, company_id=None, unknown_attempt=0, video_path=None):
        """
        Record a new attendance entry to MongoDB Cloud.
        """
        try:
            from src.utils.time_manager import time_mgr
            timestamp_str, date_str = time_mgr.get_formatted_time()
            
            # Use provided company_id or fallback to default
            cid = company_id or MongoDbConfig.COMPANY_ID
            
            # Handle image compression to WebP
            image_blob = None
            if frame is not None:
                try:
                    success, encoded_img = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
                    if success:
                        image_blob = encoded_img.tobytes()
                except Exception as e:
                    logger.error(f"MongoDB: Failed to compress image: {e}")

            # Determine status logic...
            if user_id == "Unknown":
                if status is None:
                    status = "FAILED"
            else:
                # --- COOLDOWN & IN/OUT LOGIC ---
                last_record = self.logs.find_one(
                    {"user_id": user_id, "company_id": cid, "date": date_str},
                    sort=[("_id", -1)]
                )
                
                if last_record:
                    # Check cooldown
                    last_time = last_record.get("created_at")
                    if last_time:
                        if isinstance(last_time, str):
                            try: last_time = datetime.datetime.fromisoformat(last_time)
                            except: last_time = None
                        
                        if last_time:
                            elapsed = (datetime.datetime.utcnow() - last_time).total_seconds()
                            if elapsed < RecognitionConfig.COOLDOWN_SECONDS and RecognitionConfig.COOLDOWN_SECONDS > 0:
                                logger.warning(f"MongoDB: Cooldown active for {user_name} ({int(elapsed)}s < {RecognitionConfig.COOLDOWN_SECONDS}s). Skip saving log.")
                                return last_record.get("status")
                
                # Determine IN/OUT if not provided
                if status is None:
                    status = self.get_attendance_status(user_id, company_id=cid)
                    if status == 'COOLDOWN':
                        return 'COOLDOWN'
            
            import uuid
            log_entry = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp_str,
                "date": date_str,
                "status": status,
                "shift": self.get_current_shift(),
                "company_id": cid,
                "image_webp": image_blob,
                "video_path": video_path,
                "unknown_attempt": unknown_attempt,
                "session_id": str(uuid.uuid4()),
                "uploaded_to": [],
                "created_at": datetime.datetime.utcnow()
            }
            
            self.logs.insert_one(log_entry)
            logger.info(f"Logged to MongoDB ({status}) for: {user_name} (Company: {cid})")

            # --- AUTO SYNC TO HKB ---
            self._trigger_background_sync(log_entry)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {e}")
            return None

    def _trigger_background_sync(self, log_entry):
        """
        Triggers a background thread to upload the log to HKB services.
        - Employees: Synchronized to their respective company's HKB services.
        - Strangers (Unknown): Synchronized ONLY on 5th and 10th attempt.
        """
        import threading
        
        def sync_task():
            try:
                try: 
                    from src.services.hkb_service import hkb_service
                    from src.services.report_service import report_service
                    from src.config import ApiConfig
                except Exception as e: 
                    logger.error(f"Sync Task: Import error - {e}")
                    return
                
                user_id = log_entry.get("user_id")
                cid = log_entry.get("company_id")
                attempt = log_entry.get("unknown_attempt", 0)

                # --- ALWAYS REPORT ATTENDANCE EVENT TO DASHBOARD (Even in TEST_MODE) ---
                log_id_str = str(log_entry.get("_id", ""))
                image_url = f"{ApiConfig.BASE_URL}/logs/{log_id_str}/image" if log_id_str else None
                
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
                    user_companies = self.get_user_company_ids(user_id)
                    if cid and cid not in user_companies:
                        user_companies.append(str(cid))
                    services = list(self.auth_services.find({"uuid": {"$in": user_companies}}))
                
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
                        self.mark_logs_uploaded([log_entry["_id"]], service["uuid"], details={"status": "SUCCESS", "message": res.message})
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
                            
                        self.mark_logs_uploaded([log_entry["_id"]], service["uuid"], details=details_obj)
                        logger.warning(f"Sync: Failed to sync to {service['app_name']}: {msg}")
                        
                        # CHỈ GỬI LOG KHI ĐỒNG BỘ THẤT BẠI
                        report_service.report_error(
                            message=f"Lỗi đồng bộ chấm công: {log_entry['user_name']} -> {service['app_name']}. Lỗi: {msg}",
                            status_code=502
                        )
            
            except Exception as e:
                logger.error(f"Sync: Background task error: {e}")

        threading.Thread(target=sync_task, daemon=True).start()

    def get_logs(self, company_id=None, date=None, start_date=None, end_date=None):
        """Fetch logs for a specific date, or date range, and company."""
        cid = company_id or MongoDbConfig.COMPANY_ID
        query = {"company_id": cid}
        if start_date and end_date:
            query["date"] = {"$gte": start_date, "$lte": end_date}
        elif date:
            query["date"] = date
        
        return list(self.logs.find(query).sort("_id", -1))

    def get_todays_logs(self, company_id=None):
        """Fetch logs for today for a specific company."""
        from src.utils.time_manager import time_mgr
        _, today = time_mgr.get_formatted_time()
        return self.get_logs(company_id=company_id, date=today)

    def get_log_image(self, log_id):
        """Retrieve binary image data from MongoDB."""
        from bson.objectid import ObjectId
        try:
            log = self.logs.find_one({"_id": ObjectId(log_id)})
            return log.get("image_webp") if log else None
        except Exception:
            return None

    def verify_login(self, username, password):
        """Verify login credentials with bcrypt hashing support and return role info."""
        try:
            user = self.users.find_one({"username": username})
            if user:
                stored_password = user.get("password", "")
                is_valid = False
                
                # 1. Try bcrypt verification first
                try:
                    if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                        is_valid = True
                except Exception:
                    # 2. Fallback to plain text comparison (for existing unhashed passwords)
                    if stored_password == password:
                        is_valid = True
                        # Auto-hash the password now for future security
                        self.update_user_password(username, password)
                
                if is_valid:
                    return {
                        "role": user.get("role"),
                        "company_id": user.get("company_id"),
                        "username": user.get("username"),
                        "user_id": user.get("user_id") # Trả về user_id (int) nếu có
                    }
            return None
        except Exception as e:
            logger.error(f"MongoDB: Error during login verification: {e}")
            return None

    def save_employee(self, user_id, name, birthday, company_id, sex="Nam", force_update=True, active=True):
        """
        Store or update employee metadata in MongoDB.
        Args:
            force_update: If False, will fail if user_id already exists in company.
            active: Boolean indicating if the employee is active (True) or soft-deleted (False).
        Returns: (success, message)
        """
        try:
            # 1. Validation for Null/Empty
            if not user_id or str(user_id).strip().lower() in ["none", "null", "n/a", ""]:
                return False, "Mã nhân viên không hợp lệ (Null/Empty)"
            if not name or str(name).strip().lower() in ["none", "null", "unknown", ""]:
                return False, "Tên nhân viên không được để trống"
                
            user_id = str(user_id).strip()
            company_id = str(company_id).strip()
            
            # 2. Check for duplicates if force_update is False
            if not force_update:
                exists = self.employees.find_one({"user_id": user_id, "company_id": company_id})
                if exists:
                    return False, f"Mã nhân viên {user_id} đã tồn tại trong hệ thống"

            employee_data = {
                "user_id": user_id,
                "name": name,
                "birthday": birthday,
                "company_id": company_id,
                "sex": sex,
                "active": active,
                "updated_at": datetime.datetime.utcnow()
            }
            self.employees.update_one(
                {"user_id": user_id, "company_id": company_id},
                {"$set": employee_data, "$setOnInsert": {"created_at": datetime.datetime.utcnow()}},
                upsert=True
            )
            # logger.info(f"MongoDB: Saved employee metadata for {name} (ID: {user_id})")
            return True, "Thành công"
        except Exception as e:
            logger.error(f"MongoDB: Failed to save employee: {e}")
            return False, str(e)

    def get_all_employees(self, company_id=None, active_only=False):
        """
        Get all employees from MongoDB (includes employees without face embeddings).
        """
        try:
            query = {}
            if active_only:
                query["active"] = {"$ne": False}  # True or not set (for backward compatibility)

            if company_id and company_id != "ALL":
                if isinstance(company_id, list):
                    query["company_id"] = {"$in": [str(c) for c in company_id]}
                else:
                    query["company_id"] = str(company_id)
            
            employees = list(self.employees.find(query, {"_id": 0}))
            logger.info(f"MongoDB: Found {len(employees)} employees for company_id={company_id}")
            return employees
        except Exception as e:
            logger.error(f"MongoDB: Failed to get employees: {e}")
            return []

    def get_user_company_ids(self, user_id):
        """Find all company IDs associated with this user_id from employee metadata."""
        try:
            user_id_str = str(user_id)
            cursor = self.employees.find({"user_id": user_id_str}, {"company_id": 1, "_id": 0})
            cids = list(set(str(doc["company_id"]) for doc in cursor if "company_id" in doc))
            return cids
        except Exception as e:
            logger.error(f"MongoDB: Failed to get user company IDs: {e}")
            return []

    def delete_employee(self, user_id, company_id):
        """
        Delete employee metadata and their enrollment images from MongoDB.
        """
        try:
            user_id = str(user_id).strip()
            company_id = str(company_id).strip()
            
            # 1. Delete employee metadata
            self.employees.delete_one({"user_id": user_id, "company_id": company_id})
            
            # 2. Delete enrollment images
            self.enrollment_images.delete_many({"user_id": user_id, "company_id": company_id})
            
            logger.info(f"MongoDB: Deleted employee {user_id} and their images for company {company_id}")
            return True, "Thành công"
        except Exception as e:
            logger.error(f"MongoDB: Failed to delete employee {user_id}: {e}")
            return False, str(e)

    def delete_employees_bulk(self, user_ids, company_id):
        """
        Delete multiple employees and their enrollment images from MongoDB.
        """
        try:
            if not user_ids:
                return True, "No users to delete"
                
            user_ids = [str(uid).strip() for uid in user_ids]
            company_id = str(company_id).strip()
            
            # 1. Delete employee metadata
            self.employees.delete_many({"user_id": {"$in": user_ids}, "company_id": company_id})
            
            # 2. Delete enrollment images
            self.enrollment_images.delete_many({"user_id": {"$in": user_ids}, "company_id": company_id})
            
            logger.info(f"MongoDB: Bulk deleted {len(user_ids)} employees for company {company_id}")
            return True, "Thành công"
        except Exception as e:
            logger.error(f"MongoDB: Failed to bulk delete employees: {e}")
            return False, str(e)

    def soft_delete_employee(self, user_id, company_id):
        """
        Soft delete an employee by setting active=False.
        """
        try:
            user_id = str(user_id).strip()
            company_id = str(company_id).strip()
            
            self.employees.update_one(
                {"user_id": user_id, "company_id": company_id},
                {"$set": {"active": False, "updated_at": datetime.datetime.utcnow()}}
            )
            logger.info(f"MongoDB: Soft deleted employee {user_id} for company {company_id}")
            return True, "Thành công"
        except Exception as e:
            logger.error(f"MongoDB: Failed to soft delete employee {user_id}: {e}")
            return False, str(e)

    def soft_delete_employees_bulk(self, user_ids, company_id):
        """
        Soft delete multiple employees by setting active=False.
        """
        try:
            if not user_ids:
                return True, "No users to soft delete"
                
            user_ids = [str(uid).strip() for uid in user_ids]
            company_id = str(company_id).strip()
            
            self.employees.update_many(
                {"user_id": {"$in": user_ids}, "company_id": company_id},
                {"$set": {"active": False, "updated_at": datetime.datetime.utcnow()}}
            )
            
            logger.info(f"MongoDB: Bulk soft deleted {len(user_ids)} employees for company {company_id}")
            return True, "Thành công"
        except Exception as e:
            logger.error(f"MongoDB: Failed to bulk soft delete employees: {e}")
            return False, str(e)

    # --- Management Methods ---
    
    def create_company(self, company_id, name, description=""):
        """Create or update a company record."""
        try:
            self.companies.update_one(
                {"company_id": company_id},
                {
                    "$set": {
                        "name": name,
                        "description": description,
                        "updated_at": datetime.datetime.utcnow()
                    },
                    "$setOnInsert": {"created_at": datetime.datetime.utcnow()}
                },
                upsert=True
            )
            logger.info(f"MongoDB: Created/Updated company {name} ({company_id})")
            return True, "Success"
        except Exception as e:
            logger.error(f"MongoDB: Failed to create company: {e}")
            return False, str(e)

    def create_user(self, username, password, role, company_id):
        """Create a new management user (Admin/Company)."""
        try:
            if self.users.find_one({"username": username}):
                return False, "Username already exists"
            
            # Hash the password
            hashed_pwd = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            self.users.insert_one({
                "username": username,
                "password": hashed_pwd,
                "role": role,
                "company_id": company_id,
                "created_at": datetime.datetime.utcnow()
            })
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def save_enrollment_image(self, user_id: str, company_id: str, image_blob: bytes, image_path: Optional[str] = None) -> Optional[str]:
        """
        Save an enrollment image to MongoDB.
        Returns the ObjectId of the saved image.
        """
        try:
            entry = {
                "user_id": str(user_id),
                "company_id": str(company_id),
                "image_webp": image_blob,
                "image_path": image_path, # Original path if uploaded from file
                "created_at": datetime.datetime.utcnow()
            }
            result = self.enrollment_images.insert_one(entry)
            logger.info(f"MongoDB: Saved enrollment image for user {user_id} (ID: {result.inserted_id})")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB: Failed to save enrollment image for user {user_id}: {e}")
            return None

    def get_enrollment_image(self, image_id: str) -> Optional[bytes]:
        """
        Retrieve binary enrollment image data from MongoDB.
        """
        from bson.objectid import ObjectId
        try:
            image_doc = self.enrollment_images.find_one({"_id": ObjectId(image_id)})
            return image_doc.get("image_webp") if image_doc else None
        except Exception as e:
            logger.error(f"MongoDB: Failed to get enrollment image {image_id}: {e}")
            return None

    def delete_enrollment_image(self, image_id: str) -> bool:
        """
        Delete an enrollment image from MongoDB.
        """
        from bson.objectid import ObjectId
        try:
            result = self.enrollment_images.delete_one({"_id": ObjectId(image_id)})
            if result.deleted_count > 0:
                logger.info(f"MongoDB: Deleted enrollment image {image_id}")
                return True
            return True
        except Exception as e:
            logger.error(f"MongoDB: Failed to delete enrollment image {image_id}: {e}")
            return False

    def delete_enrollment_images_by_user(self, user_id: str) -> bool:
        """
        Deletes all enrollment images associated with a specific user_id.
        """
        try:
            result = self.enrollment_images.delete_many({"user_id": user_id})
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted {result.deleted_count} enrollment images for user_id: {user_id}")
                return True
            else:
                logger.info(f"No enrollment images found for user_id: {user_id} to delete.")
                return False
        except Exception as e:
            logger.error(f"Error deleting enrollment images for user_id {user_id}: {e}")
            return False

    def get_all_companies(self):
        """Get list of all companies."""
        return list(self.companies.find().sort("name", 1))

    def get_logs_by_user(self, user_id: str, limit: Optional[int] = None, skip: int = 0) -> list:
        """
        Get all attendance logs for a specific user, with pagination support.
        """
        try:
            query = self.logs.find({"user_id": str(user_id)}).sort("timestamp", -1)
            if skip > 0:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)
            return list(query)
        except Exception as e:
            logger.error(f"MongoDB: Failed to get logs for user {user_id}: {e}")
            return []

    def delete_log(self, log_id: str) -> bool:
        """
        Delete an attendance log from MongoDB.
        """
        from bson.objectid import ObjectId
        try:
            result = self.logs.delete_one({"_id": ObjectId(log_id)})
            if result.deleted_count > 0:
                logger.info(f"MongoDB: Deleted log {log_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"MongoDB: Failed to delete log {log_id}: {e}")
            return False


    def get_company_name(self, company_id):
        """Retrieve company name from its ID (with cache)."""
        if not hasattr(self, '_company_name_cache'):
            self._company_name_cache = {}
            
        try:
            if not company_id or company_id == "admin":
                return "Admin"
            
            # Use Cache
            if company_id in self._company_name_cache:
                return self._company_name_cache[company_id]
                
            company = self.companies.find_one({"company_id": str(company_id)})
            if company:
                name = company.get("name", str(company_id))
                self._company_name_cache[company_id] = name
                return name
            return str(company_id)
        except Exception:
            return str(company_id)

    def get_all_cloud_users(self):
        """Get list of all management users."""
        return list(self.users.find().sort("username", 1))

    def update_user_password(self, username, new_password):
        """Update a management user password with hashing."""
        try:
            hashed_pwd = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            self.users.update_one(
                {"username": username},
                {"$set": {"password": hashed_pwd, "updated_at": datetime.datetime.utcnow()}}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update password for {username}: {e}")
            return False

    def get_user_by_username(self, username):
        """Find a user by username."""
        return self.users.find_one({"username": username})

    def get_user_by_email(self, email):
        """Find a user by email."""
        return self.users.find_one({"email": email})

    def delete_user(self, username):
        """Delete a management user."""
        try:
            self.users.delete_one({"username": username})
            return True
        except Exception:
            return False

    def delete_company(self, company_id):
        """Delete a company record."""
        try:
            self.companies.delete_one({"company_id": company_id})
            return True
        except Exception:
            return False

    def save_auth_service(self, data):
        """Save or update auth service connection info."""
        try:
            # data should follow src.models.auth_services.AuthService structure
            query = {"uuid": data["uuid"]}
            data["updated_at"] = datetime.datetime.utcnow()
            
            self.auth_services.update_one(
                query, 
                {"$set": data, "$setOnInsert": {"created_at": datetime.datetime.utcnow()}}, 
                upsert=True
            )
            logger.success(f"Saved auth service connection: {data.get('app_name')} ({data['uuid']})")
            return True
        except Exception as e:
            logger.error(f"Failed to save auth service: {e}")
            return False

    def set_employee_active_status(self, user_id, company_id, active=True):
        """
        Set active status for an employee in both MongoDB and Qdrant.
        """
        try:
            # 1. Update MongoDB
            res = self.employees.update_one(
                {"user_id": user_id, "company_id": company_id},
                {"$set": {"active": active, "updated_at": datetime.datetime.utcnow()}}
            )
            
            # 2. Update Qdrant
            from src.attendance.qdrant_db import attendance as qdrant_mgr
            qdrant_mgr.set_user_active_status(user_id, active)
            
            logger.info(f"MongoDB: Set active={active} for employee {user_id} (Company: {company_id})")
            return True, "Success"
        except Exception as e:
            logger.error(f"MongoDB: Failed to set active status for {user_id}: {e}")
            return False, str(e)

    def get_auth_service(self, uuid):
        """Retrieve auth service info by its UUID/SystemID."""
        try:
            return self.auth_services.find_one({"uuid": uuid})
        except Exception as e:
            logger.error(f"Failed to get auth service for {uuid}: {e}")
            return None

    def mark_logs_uploaded(self, log_ids, system_id, details=None):
        """
        Mark attendance logs as uploaded to a specific system.
        details: dict example {"status": "SUCCESS", "message": "OK"}
        """
        try:
            from bson.objectid import ObjectId
            object_ids = [ObjectId(log_id) for log_id in log_ids]
            
            update_data = {"$addToSet": {"uploaded_to": system_id}}
            if details:
                # Store full history if needed, or just last status
                details["system_id"] = system_id
                details["timestamp"] = datetime.datetime.utcnow()
                update_data["$push"] = {"upload_history": details}

            result = self.logs.update_many(
                {"_id": {"$in": object_ids}},
                update_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to mark logs as uploaded: {e}")
            return False
            
    def save_system_log(self, message, status_code, log_type="ERROR", priority="MEDIUM", devices_info=None):
        """Save a system error/info log to MongoDB for external retrieval."""
        try:
            from src.utils.time_manager import time_mgr
            vn_now = time_mgr.get_accurate_time()
            log_entry = {
                "message": message,
                "status_code": status_code,
                "type": log_type,
                "priority": priority,
                "devices_info": devices_info or {},
                "company_id": MongoDbConfig.COMPANY_ID,
                "system_id": AuthServiceConfig.SYSTEM_ID,
                "created_at": vn_now,
                "time_str": vn_now.strftime("%Y-%m-%d %H:%M:%S")
            }
            result = self.system_logs.insert_one(log_entry)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to save system log: {e}")
            return None

    def update_employee_has_face(self, user_id: str, has_face: bool = True) -> bool:
        """
        Update the has_face flag for an employee after enrollment or deletion.
        """
        try:
            self.employees.update_many(
                {"user_id": str(user_id)},
                {"$set": {"has_face": has_face, "updated_at": datetime.datetime.utcnow()}}
            )
            logger.info(f"MongoDB: Set has_face={has_face} for employee {user_id}")
            return True
        except Exception as e:
            logger.error(f"MongoDB: Failed to update has_face for {user_id}: {e}")
            return False

# Global instance
mongo_db = MongoDBManager()
