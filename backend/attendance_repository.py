import pymongo
import datetime
import bcrypt
import pytz
from typing import Optional
from loguru import logger
import os

def scoped(query, company_id):
    if company_id is not None:
        query["company_id"] = company_id
    return query


class AttendanceRepository:
    """
    Manages attendance logs and user permissions using Cloud MongoDB.
    """
    def __init__(self, db):
        self.company_id = os.getenv("COMPANY_ID", "default_company")
        self.system_id = os.getenv("SYSTEM_ID", "attendance_system")
        self.cooldown_seconds = int(os.getenv("DETECTION_COOLDOWN", "3600"))
        self.logs = db["attendance_logs"]
        self.companies = db["companies"]
        self.users = db["users"]
        self.auth_services = db["auth_services"]
        self.auth_licenses = db["auth_licenses"]
        self.settings = db["settings"]
        self.employees = db["employees"]
        self.enrollment_images = db["enrollment_images"]
        self.system_logs = db["system_logs"]
        self.system_status = db["system_status"]

    def ensure_indexes(self):
        # Create indexes for faster queries
        self.logs.create_index([("company_id", 1), ("date", -1)])
        self.logs.create_index([("user_id", 1)])
        self.logs.create_index(
            [("company_id", 1), ("device_id", 1), ("event_id", 1)], unique=True,
            partialFilterExpression={"event_id": {"$exists": True}},
        )
        self.system_status.create_index(
            [("company_id", 1), ("device_id", 1)], unique=True,
            partialFilterExpression={"device_id": {"$exists": True}},
        )
        self.users.create_index([("username", 1)], unique=True)
        self.auth_services.create_index([("uuid", 1)], unique=True)
        self.auth_licenses.create_index([("license_key", 1)], unique=True)
        self.auth_licenses.create_index([("status", 1)])
        self.auth_licenses.create_index([("expired_at", 1)])

        # Remove old unique index on key if exists, and create compound index (key, username)
        if "key_1" in self.settings.index_information():
            try:
                self.settings.drop_index("key_1")
            except pymongo.errors.OperationFailure as error:
                if error.code not in (26, 27):
                    raise
        self.settings.create_index([("key", 1), ("username", 1)], unique=True)

        self.employees.create_index([("company_id", 1), ("user_id", 1)], unique=True)
        self.enrollment_images.create_index([("user_id", 1), ("company_id", 1)])


    # --- Settings Methods ---
    def get_setting(self, key, default=None, username="GLOBAL", *, excluded_global=()):
        """Retrieve a system setting, optionally user-specific."""
        try:
            # 1. Try to find user-specific setting
            setting = self.settings.find_one({"key": key, "username": username})
            if setting:
                return setting.get("value", default)

            # 2. Fallback to GLOBAL if not found and we were looking for a specific user
            if username != "GLOBAL" and key not in excluded_global:
                setting = self.settings.find_one({"key": key, "username": "GLOBAL"})
                if setting:
                    return setting.get("value", default)

            return default
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Failed to set setting {key} for user {username}: {e}")
            return False

    def get_settings(self, keys, username, *, excluded_global=()):
        """Read effective company and global settings in a single query."""
        docs = self.settings.find({"key": {"$in": keys}, "username": {"$in": ["GLOBAL", username]}})
        by_scope = {(doc["username"], doc["key"]): doc.get("value") for doc in docs}
        effective = {key: value for (scope, key), value in by_scope.items() if scope == "GLOBAL" and key not in excluded_global}
        effective.update({key: value for (scope, key), value in by_scope.items() if scope == username})
        return effective

    def seed_setting(self, key, value, username):
        self.settings.update_one(
            {"key": key, "username": username}, {"$setOnInsert": {"value": value}}, upsert=True
        )

    def _get_cooldown(self, company_id):
        try:
            return int(self.get_setting("detection_cooldown", self.cooldown_seconds, username=company_id))
        except (TypeError, ValueError):
            return self.cooldown_seconds

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
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception:
            return "Other"

    def get_attendance_status(self, user_id, company_id=None, *, captured_at=None):
        """Checks the last log to determine if user should be IN, OUT or COOLDOWN with Shift Support."""
        try:
            from src.utils.time_manager import time_mgr
            event_time = captured_at or datetime.datetime.now(pytz.utc)
            utc_time = event_time.astimezone(pytz.utc).replace(tzinfo=None)
            vn_now = event_time.astimezone(pytz.timezone("Asia/Ho_Chi_Minh")) if captured_at else time_mgr.get_accurate_time()
            date_str = vn_now.strftime("%Y-%m-%d")
            cid = company_id or self.company_id

            cooldown = self._get_cooldown(cid)

            # Current time in VN (HHMM) for shift logic
            current_time_val = vn_now.hour * 100 + vn_now.minute

            # 1. Fetch last record today
            day_filter = {"user_id": user_id, "company_id": cid, "date": date_str}
            if captured_at:
                day_filter["created_at"] = {"$lte": utc_time}
            last_record = self.logs.find_one(day_filter, sort=[("created_at", -1), ("_id", -1)])

            # 2. Check Cooldown
            if last_record:
                last_time = last_record.get("created_at")
                if last_time:
                    if isinstance(last_time, str):
                        try: last_time = datetime.datetime.fromisoformat(last_time)
                        except: last_time = None

                    if last_time:
                        elapsed = (utc_time - last_time).total_seconds()
                        if elapsed < cooldown:
                            return 'COOLDOWN'

            # 3. Shift Logic
            # Ca sáng: 00:05 - 12:59
            if 5 <= current_time_val <= 1259:
                morning_in = self.logs.find_one({
                    **day_filter, "status": "IN",
                })
                if not morning_in:
                    return 'IN'

            # Ca chiều: 13:00 - 23:59
            elif 1300 <= current_time_val <= 2359:
                # Define start of afternoon shift in UTC
                vn_afternoon_start = vn_now.replace(hour=13, minute=0, second=0, microsecond=0)
                utc_afternoon_start = vn_afternoon_start.astimezone(pytz.utc).replace(tzinfo=None)

                afternoon_in = self.logs.find_one({
                    **day_filter, "status": "IN",
                    "created_at": {**day_filter.get("created_at", {}), "$gte": utc_afternoon_start},
                })
                if not afternoon_in:
                    return 'IN'

            # 4. Default: Toggle IN/OUT based on last record
            if last_record and last_record.get('status') == 'IN':
                return 'OUT'

            return 'IN'
        except pymongo.errors.PyMongoError:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Error getting status: {e}")
            return 'IN'

    def log_attendance(self, user_id, user_name, status=None, image_blob=None, company_id=None,
                       unknown_attempt=0, video_path=None, *, device_id=None, event_id=None, captured_at=None):
        """
        Record a new attendance entry to MongoDB Cloud.
        """
        try:
            from src.utils.time_manager import time_mgr
            cid = company_id or self.company_id
            received_at = datetime.datetime.utcnow()
            if captured_at:
                vn_now = captured_at.astimezone(pytz.timezone("Asia/Ho_Chi_Minh"))
                timestamp_str, date_str = vn_now.strftime("%Y-%m-%d %H:%M:%S"), vn_now.strftime("%Y-%m-%d")
                hour = vn_now.hour * 100 + vn_now.minute
                shift = "Afternoon" if hour >= 1300 else "Morning" if hour >= 5 else "Other"
            else:
                timestamp_str, date_str = time_mgr.get_formatted_time()
                shift = self.get_current_shift()

            identity = {"company_id": cid, "device_id": device_id, "event_id": event_id} if event_id else None
            if identity:
                existing = self.logs.find_one(identity, {"image_webp": 0})
                if existing:
                    return existing["status"], existing

            # Device events use the time and shift of capture, including queued events.
            if captured_at and user_id not in {"Unknown", "Spoof"}:
                status = self.get_attendance_status(user_id, company_id=cid, captured_at=captured_at)
                if status == "COOLDOWN":
                    return status, None
            elif user_id in {"Unknown", "Spoof"}:
                if status is None:
                    status = "FAILED"
            else:
                cooldown = self._get_cooldown(cid)
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
                            if elapsed < cooldown and cooldown > 0:
                                logger.warning(f"MongoDB: Cooldown active for {user_name} ({int(elapsed)}s < {cooldown}s). Skip saving log.")
                                return last_record.get("status"), None

                # Determine IN/OUT if not provided
                if status is None:
                    status = self.get_attendance_status(user_id, company_id=cid)
                    if status == 'COOLDOWN':
                        return 'COOLDOWN', None

            import uuid
            log_entry = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp_str,
                "date": date_str,
                "status": status,
                "shift": shift,
                "company_id": cid,
                "image_webp": image_blob,
                "video_path": video_path,
                "unknown_attempt": unknown_attempt,
                "session_id": str(uuid.uuid4()),
                "uploaded_to": [],
                "created_at": captured_at.astimezone(pytz.utc).replace(tzinfo=None) if captured_at else received_at,
            }
            if device_id:
                log_entry.update(device_id=device_id, received_at=received_at)
            if event_id:
                log_entry["event_id"] = event_id

            try:
                self.logs.insert_one(log_entry)
            except pymongo.errors.DuplicateKeyError:
                if identity:
                    existing = self.logs.find_one(identity, {"image_webp": 0})
                    if existing:
                        return existing["status"], existing
                raise
            logger.info(f"Logged to MongoDB ({status}) for: {user_name} (Company: {cid})")

            # The calling camera service runs its existing HKB/dashboard sync.
            del log_entry["image_webp"]
            return status, log_entry

        except pymongo.errors.PyMongoError:
            raise
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {e}")
            return None, None

    def get_logs(self, company_id=None, date=None, start_date=None, end_date=None):
        """Fetch logs for a specific date, or date range, and company."""
        cid = company_id or self.company_id
        query = {"company_id": cid}
        if start_date and end_date:
            query["date"] = {"$gte": start_date, "$lte": end_date}
        elif date:
            query["date"] = date

        return list(self.logs.find(query, {"image_webp": 0}).sort("_id", -1))

    def get_todays_logs(self, company_id=None):
        """Fetch logs for today for a specific company."""
        from src.utils.time_manager import time_mgr
        _, today = time_mgr.get_formatted_time()
        return self.get_logs(company_id=company_id, date=today)

    def get_log_image(self, log_id, *, company_id=None):
        """Retrieve binary image data from MongoDB."""
        from bson.objectid import ObjectId
        try:
            log = self.logs.find_one(scoped({"_id": ObjectId(log_id)}, company_id))
            return log.get("image_webp") if log else None
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to get employees: {e}")
            return []

    def get_user_company_ids(self, user_id, *, company_id=None):
        """Find all company IDs associated with this user_id from employee metadata."""
        try:
            user_id_str = str(user_id)
            cursor = self.employees.find(scoped({"user_id": user_id_str}, company_id), {"company_id": 1, "_id": 0})
            cids = list(set(str(doc["company_id"]) for doc in cursor if "company_id" in doc))
            return cids
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to save enrollment image for user {user_id}: {e}")
            return None

    def get_enrollment_image(self, image_id: str, *, company_id=None) -> Optional[bytes]:
        """
        Retrieve binary enrollment image data from MongoDB.
        """
        from bson.objectid import ObjectId
        try:
            image_doc = self.enrollment_images.find_one(scoped({"_id": ObjectId(image_id)}, company_id))
            return image_doc.get("image_webp") if image_doc else None
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to get enrollment image {image_id}: {e}")
            return None

    def delete_enrollment_image(self, image_id: str, *, company_id=None) -> bool:
        """
        Delete an enrollment image from MongoDB.
        """
        from bson.objectid import ObjectId
        try:
            result = self.enrollment_images.delete_one(scoped({"_id": ObjectId(image_id)}, company_id))
            if result.deleted_count > 0:
                logger.info(f"MongoDB: Deleted enrollment image {image_id}")
                return True
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to delete enrollment image {image_id}: {e}")
            return False

    def delete_enrollment_images_by_user(self, user_id: str, *, company_id=None) -> bool:
        """
        Deletes all enrollment images associated with a specific user_id.
        """
        try:
            result = self.enrollment_images.delete_many(scoped({"user_id": user_id}, company_id))
            if result.deleted_count > 0:
                logger.info(f"Successfully deleted {result.deleted_count} enrollment images for user_id: {user_id}")
                return True
            else:
                logger.info(f"No enrollment images found for user_id: {user_id} to delete.")
                return False
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Error deleting enrollment images for user_id {user_id}: {e}")
            return False

    def get_all_companies(self):
        """Get list of all companies."""
        return list(self.companies.find().sort("name", 1))

    def get_logs_by_user(self, user_id: str, limit: Optional[int] = None, skip: int = 0, *, company_id=None) -> list:
        """
        Get all attendance logs for a specific user, with pagination support.
        """
        try:
            query = self.logs.find(scoped({"user_id": str(user_id)}, company_id)).sort("timestamp", -1)
            if skip > 0:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)
            return list(query)
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to get logs for user {user_id}: {e}")
            return []

    def delete_log(self, log_id: str, *, company_id=None) -> bool:
        """
        Delete an attendance log from MongoDB.
        """
        from bson.objectid import ObjectId
        try:
            result = self.logs.delete_one(scoped({"_id": ObjectId(log_id)}, company_id))
            if result.deleted_count > 0:
                logger.info(f"MongoDB: Deleted log {log_id}")
                return True
            return False
        except pymongo.errors.ConnectionFailure:
            raise
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
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception:
            return str(company_id)

    def get_all_cloud_users(self):
        """Get list of all management users."""
        return list(self.users.find({}, {"password": 0}).sort("username", 1))

    def update_user_password(self, username, new_password):
        """Update a management user password with hashing."""
        try:
            hashed_pwd = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            self.users.update_one(
                {"username": username},
                {"$set": {"password": hashed_pwd, "updated_at": datetime.datetime.utcnow()}, "$inc": {"auth_version": 1}}
            )
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Failed to update password for {username}: {e}")
            return False

    def get_user_by_username(self, username):
        """Find a user by username."""
        return self.users.find_one({"username": username}, {"password": 0})

    def get_user_by_email(self, email):
        """Find a user by email."""
        return self.users.find_one({"email": email}, {"password": 0})

    def delete_user(self, username):
        """Delete a management user."""
        try:
            self.users.delete_one({"username": username})
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception:
            return False

    def delete_company(self, company_id):
        """Delete a company record."""
        try:
            self.companies.delete_one({"company_id": company_id})
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception:
            return False

    def save_auth_service(self, data, *, owner_filter=None):
        """Save or update auth service connection info."""
        try:
            # data should follow src.models.auth_services.AuthService structure
            query = {"uuid": data["uuid"]}
            if owner_filter:
                query = {"$and": [query, owner_filter]}
            data["updated_at"] = datetime.datetime.utcnow()

            self.auth_services.update_one(
                query,
                {"$set": data, "$setOnInsert": {"created_at": datetime.datetime.utcnow()}},
                upsert=True
            )
            logger.success(f"Saved auth service connection: {data.get('app_name')} ({data['uuid']})")
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Failed to save auth service: {e}")
            return False

    def set_employee_active_status(self, user_id, company_id, active=True):
        """
        Set active status for an employee in MongoDB.
        """
        try:
            self.employees.update_one(
                {"user_id": user_id, "company_id": company_id},
                {"$set": {"active": active, "updated_at": datetime.datetime.utcnow()}}
            )


            logger.info(f"MongoDB: Set active={active} for employee {user_id} (Company: {company_id})")
            return True, "Success"
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to set active status for {user_id}: {e}")
            return False, str(e)

    def get_auth_service(self, uuid):
        """Retrieve auth service info by its UUID/SystemID."""
        try:
            return self.auth_services.find_one({"uuid": uuid})
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Failed to get auth service for {uuid}: {e}")
            return None

    def mark_logs_uploaded(self, log_ids, system_id, details=None, *, company_id=None, device_id=None):
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

            query = scoped({"_id": {"$in": object_ids}}, company_id)
            if device_id:
                query["device_id"] = device_id
            self.logs.update_many(query, update_data)
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Failed to mark logs as uploaded: {e}")
            return False

    def save_system_log(self, message, status_code, log_type="ERROR", priority="MEDIUM", devices_info=None, *, company_id=None, system_id=None):
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
                "company_id": company_id or self.company_id,
                "system_id": system_id or self.system_id,
                "created_at": vn_now,
                "time_str": vn_now.strftime("%Y-%m-%d %H:%M:%S")
            }
            result = self.system_logs.insert_one(log_entry)
            return str(result.inserted_id)
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"Failed to save system log: {e}")
            return None

    def update_employee_has_face(self, user_id: str, has_face: bool = True, *, company_id=None) -> bool:
        """
        Update the has_face flag for an employee after enrollment or deletion.
        """
        try:
            self.employees.update_many(
                scoped({"user_id": str(user_id)}, company_id),
                {"$set": {"has_face": has_face, "updated_at": datetime.datetime.utcnow()}}
            )
            logger.info(f"MongoDB: Set has_face={has_face} for employee {user_id}")
            return True
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to update has_face for {user_id}: {e}")
            return False

    # --- License Repository Methods (Bittech Auth SDK) ---

    def find_latest_license(self, system_id: Optional[str] = None, *, company_id=None) -> Optional[dict]:
        """
        Lấy bản ghi license mới nhất từ MongoDB.
        """
        try:
            query = {}
            if system_id:
                query["system_id"] = system_id
            return self.auth_licenses.find_one(scoped(query, company_id), sort=[("created_at", pymongo.DESCENDING)])
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to find latest license: {e}")
            return None

    def find_active_license(self, now: Optional[datetime.datetime] = None, system_id: Optional[str] = None, *, company_id=None) -> Optional[dict]:
        """
        Lấy license đang ở trạng thái ACTIVE (status = 1) và chưa hết hạn.
        """
        try:
            if now is None:
                now = datetime.datetime.utcnow()

            query = {
                "status": 1,
                "expired_at": {"$gt": now}
            }
            if system_id:
                query["system_id"] = system_id

            return self.auth_licenses.find_one(scoped(query, company_id), sort=[("created_at", pymongo.DESCENDING)])
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to find active license: {e}")
            return None

    def save_pending_license(
        self,
        *,
        public_id: str,
        license_key: str,
        license_seed: str,
        label: str,
        expired_at: datetime.datetime,
        signature: Optional[str] = None,
        notes: Optional[str] = None,
        system_id: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> dict:
        """
        Lưu hoặc cập nhật bản ghi license ở trạng thái PENDING (status = 0).
        """
        now = datetime.datetime.utcnow()
        doc = {
            "public_id": str(public_id),
            "license_key": str(license_key),
            "license_seed": str(license_seed),
            "label": str(label),
            "status": 0,
            "signature": signature,
            "expired_at": expired_at,
            "verified_at": None,
            "notes": notes or "",
            "system_id": system_id or self.system_id,
            "company_id": company_id or self.company_id,
            "updated_at": now,
        }
        self.auth_licenses.update_one(
            {"license_key": license_key},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": now}
            },
            upsert=True
        )
        logger.info(f"MongoDB: Saved pending license {license_key} (label: {label})")
        return doc

    def mark_license_verified(
        self,
        license_key: str,
        verified_at: Optional[datetime.datetime] = None,
        signature: Optional[str] = None,
    ) -> bool:
        """
        Đánh dấu license đã được verify thành công (status = 1).
        """
        try:
            now = verified_at or datetime.datetime.utcnow()
            update_fields = {
                "status": 1,
                "verified_at": now,
                "updated_at": now,
            }
            if signature:
                update_fields["signature"] = signature

            result = self.auth_licenses.update_one(
                {"license_key": license_key},
                {"$set": update_fields}
            )
            if result.matched_count > 0:
                logger.success(f"MongoDB: License {license_key} marked as ACTIVE (status=1)")
                return True
            return False
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to mark license {license_key} verified: {e}")
            return False

    def update_license_status(self, license_key: str, status: int) -> bool:
        """
        Cập nhật trạng thái của license (0: Pending, 1: Active, 2: Expired/Revoked).
        """
        try:
            result = self.auth_licenses.update_one(
                {"license_key": license_key},
                {
                    "$set": {
                        "status": status,
                        "updated_at": datetime.datetime.utcnow()
                    }
                }
            )
            return result.matched_count > 0
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to update license status {license_key}: {e}")
            return False

    def get_unuploaded_logs(self, system_id: str, limit: int = 50, *, company_id=None) -> list:
        """
        Lấy danh sách các bản ghi chấm công chưa được đồng bộ lên system_id (để sync offline/bù).
        """
        try:
            query = {
                "uploaded_to": {"$ne": system_id},
                "user_id": {"$nin": ["Unknown", "Spoof"]}
            }
            return list(self.logs.find(scoped(query, company_id)).sort("timestamp", pymongo.ASCENDING).limit(limit))
        except pymongo.errors.ConnectionFailure:
            raise
        except Exception as e:
            logger.error(f"MongoDB: Failed to get unuploaded logs for system {system_id}: {e}")
            return []

    def get_employee(self, user_id, company_id=None):
        query = {"user_id": str(user_id)}
        if company_id is not None:
            query["company_id"] = company_id
        return self.employees.find_one(query)

    def get_company(self, company_id):
        return self.companies.find_one({"company_id": company_id})

    def get_auth_services_for_user(self, user_id):
        return list(self.auth_services.find({"user_id": user_id}))

    def get_auth_services_for_companies(self, company_ids):
        return list(self.auth_services.find({"uuid": {"$in": company_ids}}))

    def delete_auth_service(self, system_id):
        self.auth_services.delete_one({"uuid": system_id})

    def get_camera_status(self, company_id):
        return self.system_status.find_one(
            {"type": "camera_service", "company_id": company_id}, sort=[("last_seen", -1)],
        )

    def update_camera_status(self, company_id, last_seen, status, camera_connected, *, device_id=None):
        query = {"type": "camera_service", "company_id": company_id}
        if device_id:
            query["device_id"] = device_id
        self.system_status.update_one(
            query,
            {"$set": {"last_seen": last_seen, "status": status, "camera_connected": camera_connected}},
            upsert=True,
        )

    def get_system_logs(self, date=None, *, company_id=None):
        query = {}
        if date:
            start = datetime.datetime.strptime(date, "%Y-%m-%d")
            day = {"$gte": start, "$lt": start + datetime.timedelta(days=1)}
            query = {"$or": [
                {"time_str": {"$regex": "^" + start.strftime("%Y-%m-%d")}},
                {"created_at": day},
                {"timestamp": day},
            ]}
        return list(self.system_logs.find(scoped(query, company_id)).sort("_id", -1).limit(200))

    def record_system_log(self, timestamp, time_str, level, message, source, *, company_id=None, device_id=None):
        self.system_logs.insert_one({
            "timestamp": timestamp, "time_str": time_str, "level": level,
            "message": message, "source": source, "company_id": company_id, "device_id": device_id,
        })
