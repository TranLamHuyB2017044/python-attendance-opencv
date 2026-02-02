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
                    image_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"SQLite Attendance DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite DB: {e}")

    def log_attendance(self, user_id, user_name, image_path=None):
        """
        Record a new attendance entry using anti-cheat time.
        """
        try:
            from src.utils.time_manager import time_mgr
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp, date = time_mgr.get_formatted_time()
            
            cursor.execute('''
                INSERT INTO attendance_logs (user_id, user_name, timestamp, date, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, user_name, timestamp, date, image_path))
            
            conn.commit()
            conn.close()
            logger.info(f"Logged attendance in SQLite for: {user_name} ({user_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to log attendance to SQLite: {e}")
            return False

    def get_todays_logs(self):
        """Retrieve all logs for the current day."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute('SELECT * FROM attendance_logs WHERE date = ?', (today,))
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Error fetching logs: {e}")
            return []

# Global instance
db = AttendanceDB()
