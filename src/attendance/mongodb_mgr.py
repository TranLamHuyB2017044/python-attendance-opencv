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

            log_entry = {
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": timestamp_str,
                "date": date_str,
                "status": status,
                "company_id": cid,
                "image_webp": image_blob,
                "created_at": datetime.utcnow()
            }
            
            self.logs.insert_one(log_entry)
            logger.info(f"Logged to MongoDB ({status}) for: {user_name} (Company: {cid})")
            return status
            
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {e}")
            return None

    def get_todays_logs(self, company_id=None):
        """Fetch logs for today for a specific company."""
        from src.utils.time_manager import time_mgr
        _, today = time_mgr.get_formatted_time()
        
        cid = company_id or MongoDbConfig.COMPANY_ID
        query = {"date": today, "company_id": cid}
        return list(self.logs.find(query).sort("_id", -1))

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

    # --- Management Methods ---
    
    def create_company(self, company_id, name, description=""):
        """Create a new company record."""
        try:
            if self.companies.find_one({"company_id": company_id}):
                return False, "Company ID already exists"
            self.companies.insert_one({
                "company_id": company_id,
                "name": name,
                "description": description,
                "created_at": datetime.utcnow()
            })
            return True, "Success"
        except Exception as e:
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

# Global instance
mongo_db = MongoDBManager()
