import pymongo
from datetime import datetime
from loguru import logger
import cv2
import numpy as np
from src.config import MongoDbConfig

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
            self.settings.create_index([("key", 1)], unique=True)
            self.employees.create_index([("company_id", 1), ("user_id", 1)], unique=True)
            
            logger.info("Connected to MongoDB Cloud successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")

    # --- Settings Methods ---
    def get_setting(self, key, default=None):
        """Retrieve a system setting."""
        try:
            setting = self.settings.find_one({"key": key})
            return setting.get("value", default) if setting else default
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return default

    def set_setting(self, key, value):
        """Save/Update a system setting."""
        try:
            self.settings.update_one(
                {"key": key},
                {"$set": {"value": value, "updated_at": datetime.utcnow()}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set setting {key}: {e}")
            return False

    def log_attendance(self, user_id, user_name, status=None, frame=None, company_id=None):
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

            # Determine IN/OUT status if not provided (Local logic shifted to Cloud context)
            if status is None:
                if user_id == "Unknown":
                    status = "FAILED"
                else:
                    last_record = self.logs.find_one(
                        {"user_id": user_id, "date": date_str, "company_id": cid},
                        sort=[("_id", -1)]
                    )
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
                "session_id": str(uuid.uuid4()),  # Unique ID for this attendance record
                "uploaded_to": [],  # List of system_ids this has been uploaded to
                "created_at": datetime.utcnow()
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
        - Strangers (Unknown): Synchronized to ALL connected HKB services (Global alert).
        """
        import threading
        
        def sync_task():
            try:
                from src.services.hkb_service import hkb_service
                
                user_id = log_entry.get("user_id")
                cid = log_entry.get("company_id")
                
                if user_id == "Unknown":
                    # 1. Nếu là người lạ, lấy tất cả các dịch vụ HKB đã kết nối trong hệ thống
                    logger.info("Sync: Stranger detected! Broadcasting to all connected HKB systems...")
                    services = list(self.auth_services.find({}))
                else:
                    # 2. Nếu là nhân viên, chỉ gửi tới dịch vụ của công ty đó
                    services = list(self.auth_services.find({"uuid": cid}))
                
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
                        self.mark_logs_uploaded([log_entry["_id"]], service["uuid"])
                        logger.success(f"Sync: Successfully synced to {service['app_name']}")
                    else:
                        logger.warning(f"Sync: Failed to sync to {service['app_name']}: {res.message if res else 'No response'}")
            
            except Exception as e:
                logger.error(f"Sync: Background task error: {e}")

        threading.Thread(target=sync_task, daemon=True).start()

    def get_logs(self, company_id=None, date=None):
        """Fetch logs for a specific date and company."""
        cid = company_id or MongoDbConfig.COMPANY_ID
        query = {"company_id": cid}
        if date:
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

    def save_employee(self, user_id, name, birthday, company_id):
        """
        Store or update employee metadata in MongoDB.
        """
        try:
            # Ensure user_id is string for consistency
            user_id = str(user_id)
            company_id = str(company_id)
            
            employee_data = {
                "user_id": user_id,
                "name": name,
                "birthday": birthday,
                "company_id": company_id,
                "updated_at": datetime.utcnow()
            }
            self.employees.update_one(
                {"user_id": user_id, "company_id": company_id},
                {"$set": employee_data, "$setOnInsert": {"created_at": datetime.utcnow()}},
                upsert=True
            )
            logger.info(f"MongoDB: Saved employee metadata for {name} (ID: {user_id})")
            return True
        except Exception as e:
            logger.error(f"MongoDB: Failed to save employee: {e}")
            return False

    def get_all_employees(self, company_id=None):
        """
        Get all employees from MongoDB (includes employees without face embeddings).
        
        Args:
            company_id: Optional company filter
            
        Returns:
            List of employee dicts with user_id, name, birthday
        """
        try:
            query = {}
            if company_id:
                query["company_id"] = str(company_id)  # Ensure string for consistency
            
            employees = list(self.employees.find(query, {"_id": 0, "user_id": 1, "name": 1, "birthday": 1}))
            return employees
        except Exception as e:
            logger.error(f"MongoDB: Failed to get employees: {e}")
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
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {"created_at": datetime.utcnow()}
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
                "created_at": datetime.utcnow()
            })
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def get_all_companies(self):
        """Get list of all companies."""
        return list(self.companies.find().sort("name", 1))

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
            data["updated_at"] = datetime.utcnow()
            
            self.auth_services.update_one(
                query, 
                {"$set": data, "$setOnInsert": {"created_at": datetime.utcnow()}}, 
                upsert=True
            )
            logger.success(f"Saved auth service connection: {data.get('app_name')} ({data['uuid']})")
            return True
        except Exception as e:
            logger.error(f"Failed to save auth service: {e}")
            return False

    def mark_logs_uploaded(self, log_ids, system_id):
        """Mark attendance logs as uploaded to a specific system."""
        try:
            from bson.objectid import ObjectId
            object_ids = [ObjectId(log_id) for log_id in log_ids]
            
            result = self.logs.update_many(
                {"_id": {"$in": object_ids}},
                {"$addToSet": {"uploaded_to": system_id}}
            )
            logger.success(f"Marked {result.modified_count} logs as uploaded to {system_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark logs as uploaded: {e}")
            return False

# Global instance
mongo_db = MongoDBManager()
