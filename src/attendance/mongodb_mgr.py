import pymongo
import datetime
from loguru import logger
import cv2
import numpy as np
from src.config import MongoDbConfig, RecognitionConfig

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

    def get_attendance_status(self, user_id, company_id=None):
        """Checks the last log to determine if user should be IN, OUT or COOLDOWN."""
        try:
            from src.utils.time_manager import time_mgr
            _, date_str = time_mgr.get_formatted_time()
            cid = company_id or MongoDbConfig.COMPANY_ID
            
            last_record = self.logs.find_one(
                {"user_id": user_id, "company_id": cid, "date": date_str},
                sort=[("_id", -1)]
            )
            
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
                
                # Alternate IN/OUT
                return 'OUT' if last_record.get('status') == 'IN' else 'IN'
            
            return 'IN' # First time today
        except Exception as e:
            logger.error(f"MongoDB: Error getting status: {e}")
            return 'IN'

    def log_attendance(self, user_id, user_name, status=None, frame=None, company_id=None, unknown_attempt=0):
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
                            if elapsed < RecognitionConfig.COOLDOWN_SECONDS:
                                logger.warning(f"MongoDB: Cooldown active for {user_name} ({int(elapsed)}s < {RecognitionConfig.COOLDOWN_SECONDS}s). Skip saving log.")
                                return last_record.get("status")
                
                # Determine IN/OUT if not provided
                if status is None:
                    if not last_record or last_record.get('status') in ['OUT', 'FAILED']:
                        status = 'IN'
                    else:
                        status = 'OUT'
            
            import uuid
            log_entry = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp_str,
                "date": date_str,
                "status": status,
                "company_id": cid,
                "image_webp": image_blob,
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
                try: from src.services.hkb_service import hkb_service
                except ImportError: return
                
                user_id = log_entry.get("user_id")
                cid = log_entry.get("company_id")
                attempt = log_entry.get("unknown_attempt", 0)
                
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
                    else:
                        msg = res.message if res else 'No response'
                        self.mark_logs_uploaded([log_entry["_id"]], service["uuid"], details={"status": "FAILED", "message": msg})
                        logger.warning(f"Sync: Failed to sync to {service['app_name']}: {msg}")
            
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
        """Verify user login and return role info."""
        user = self.users.find_one({"username": username, "password": password})
        if user:
            return {
                "role": user.get("role"),
                "company_id": user.get("company_id"),
                "username": user.get("username"),
                "user_id": user.get("user_id") # Trả về user_id (int) nếu có
            }
        return None

    def save_employee(self, user_id, name, birthday, company_id, force_update=True):
        """
        Store or update employee metadata in MongoDB.
        Args:
            force_update: If False, will fail if user_id already exists in company.
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

    def get_all_employees(self, company_id=None):
        """
        Get all employees from MongoDB (includes employees without face embeddings).
        """
        try:
            query = {}
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
            self.users.insert_one({
                "username": username,
                "password": password,
                "role": role,
                "company_id": company_id,
                "created_at": datetime.datetime.utcnow()
            })
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def get_all_companies(self):
        """Get list of all companies."""
        return list(self.companies.find().sort("name", 1))

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

# Global instance
mongo_db = MongoDBManager()
