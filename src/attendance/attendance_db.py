import sqlite3
from datetime import datetime
from loguru import logger
from src.config import DATA_DIR

class AttendanceDB:
    """
    Manages attendance logs using a local SQLite database.
    """
    def __init__(self, db_name="attendance.db"):
        self.db_path = DATA_DIR / db_name
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table for attendance logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT,
                    image_path TEXT,
                    image_data BLOB
                )
            ''')
            
            # Check if status column exists, if not add it (for existing databases)
            cursor.execute("PRAGMA table_info(attendance_logs)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'status' not in columns:
                cursor.execute("ALTER TABLE attendance_logs ADD COLUMN status TEXT")
                logger.info("Added 'status' column to attendance_logs table")
            
            if 'image_path' not in columns:
                cursor.execute("ALTER TABLE attendance_logs ADD COLUMN image_path TEXT")
                logger.info("Added 'image_path' column to attendance_logs table")

            if 'image_data' not in columns:
                cursor.execute("ALTER TABLE attendance_logs ADD COLUMN image_data BLOB")
                logger.info("Added 'image_data' column to attendance_logs table")

            # Table for admin users
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admin_users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL
                )
            ''')
            
            # Create a default admin if none exists
            cursor.execute('SELECT COUNT(*) FROM admin_users')
            if cursor.fetchone()[0] == 0:
                # Default credentials: admin / 123456
                cursor.execute('INSERT INTO admin_users (username, password) VALUES (?, ?)', 
                             ('admin', '123456'))
                logger.info("Default admin user created: admin / 123456")

            conn.commit()
            conn.close()
            logger.info(f"SQLite Attendance DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")

    def log_attendance(self, user_id, user_name, image_path=None, status=None, frame=None):
        """
        Record a new attendance entry using anti-cheat time.
        """
        try:
            import cv2
            from src.utils.time_manager import time_mgr
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp, date = time_mgr.get_formatted_time()
            
            # Handle image compression if frame is provided
            image_blob = None
            if frame is not None:
                try:
                    # Compress to WebP with 75% quality for better compression than JPEG
                    success, encoded_img = cv2.imencode('.webp', frame, [int(cv2.IMWRITE_WEBP_QUALITY), 75])
                    if success:
                        image_blob = encoded_img.tobytes()
                except Exception as e:
                    logger.error(f"Failed to compress image to WebP: {e}")
            
            # Determine IN/OUT status if not provided
            if status is not None:
                current_status = status
            elif user_id == "Unknown":
                current_status = "FAILED"
            else:
                # Find the last record for this user today
                cursor.execute('''
                    SELECT status FROM attendance_logs 
                    WHERE user_id = ? AND date = ? 
                    ORDER BY id DESC LIMIT 1
                ''', (user_id, date))
                
                last_record = cursor.fetchone()
                
                if last_record is None or last_record[0] == 'OUT' or last_record[0] == 'FAILED':
                    current_status = 'IN'
                else:
                    current_status = 'OUT'
            
            cursor.execute('''
                INSERT INTO attendance_logs (user_id, user_name, timestamp, date, status, image_path, image_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, user_name, timestamp, date, current_status, image_path, image_blob))
            
            conn.commit()
            conn.close()
            logger.info(f"Logged attendance ({current_status}) in SQLite for: {user_name} ({user_id})")
            return current_status
        except Exception as e:
            logger.error(f"Failed to log attendance to SQLite: {e}")
            return None

    def get_todays_logs(self):
        """Retrieve all logs for the current day."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute('SELECT * FROM attendance_logs WHERE date = ? ORDER BY id DESC', (today,))
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Error fetching logs: {e}")
            return []

    def get_all_logs(self, limit=100):
        """Retrieve all logs with limit."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row # To access by column name
            cursor = conn.cursor()
            cursor.execute('SELECT id, user_id, user_name, timestamp, date, status, image_path FROM attendance_logs ORDER BY id DESC LIMIT ?', (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Error fetching all logs: {e}")
            return []

    def get_log_image(self, log_id):
        """Retrieve the binary image data for a specific log."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT image_data FROM attendance_logs WHERE id = ?', (log_id,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error fetching log image: {e}")
            return None

    def verify_admin(self, username, password):
        """Verify admin credentials."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM admin_users WHERE username = ? AND password = ?', (username, password))
            admin = cursor.fetchone()
            conn.close()
            return admin is not None
        except Exception as e:
            logger.error(f"Error verifying admin: {e}")
            return False

# Global instance
db = AttendanceDB()
