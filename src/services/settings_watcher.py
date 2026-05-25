import time
import threading
from loguru import logger
from pymongo.errors import PyMongoError

from src.config import CameraConfig
from src.attendance.mongodb_mgr import mongo_db

class SettingsWatcher:
    """
    Background worker that listens to changes in the settings collection 
    using MongoDB Change Streams. If Change Streams are not supported 
    (e.g., in a standalone local MongoDB deployment), it automatically 
    falls back to a non-blocking 30-second background polling mechanism.
    """
    def __init__(self, camera_instance):
        self.camera = camera_instance
        self.running = False
        self.thread = None

    def start(self):
        """Start the background settings watcher daemon thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="settings_watcher")
        self.thread.start()
        logger.info("SettingsWatcher background thread started.")

    def stop(self):
        """Stop the background settings watcher thread."""
        self.running = False
        logger.info("SettingsWatcher background thread stopped.")

    def _run(self):
        # Try to use MongoDB Change Streams first
        try:
            logger.info("Attempting to initialize MongoDB Change Stream for settings...")
            pipeline = [
                {
                    "$match": {
                        "operationType": {"$in": ["insert", "update", "replace", "delete"]}
                    }
                }
            ]
            # Listen to changes in the 'settings' collection with update lookup
            with mongo_db.settings.watch(pipeline, full_document="updateLookup") as stream:
                logger.success("MongoDB Change Stream successfully initialized. Watching settings...")
                while self.running:
                    # Non-blocking pull with timeout to allow thread shutdown/yield
                    change = stream.try_next()
                    if change is not None:
                        # Extract the full document to check key and company ID
                        doc = change.get("fullDocument")
                        if doc:
                            key = doc.get("key")
                            username = doc.get("username")
                            
                            from src.config import MongoDbConfig
                            current_company = MongoDbConfig.COMPANY_ID
                            
                            watched_keys = {
                                "camera_ip", "camera_port", "camera_user", "camera_pass",
                                "camera_rtsp_path",
                                "camera_roi", "detection_cooldown", "anti_spoofing_enabled",
                                "enable_telegram_notif", "recognition_threshold", "gather_frames"
                            }
                            
                            if username in [current_company, "GLOBAL"] and key in watched_keys:
                                logger.info(f"MongoDB setting change detected for key '{key}' (company: '{username}'). Triggering configuration reload.")
                                self._reload_and_update()
                            else:
                                logger.debug(f"Ignoring irrelevant settings change for key '{key}', company '{username}'.")
                        else:
                            # Fallback if fullDocument is not available (e.g. on older MongoDB versions or delete operations)
                            logger.info("MongoDB setting change detected (no fullDocument details). Triggering configuration reload.")
                            self._reload_and_update()
                    time.sleep(0.5)
        except PyMongoError as e:
            logger.warning(
                f"MongoDB Change Streams not supported or failed to start: {e}. "
                "Switching to robust background polling (every 30 seconds)..."
            )
            self._run_polling()
        except Exception as e:
            logger.warning(
                f"Unexpected error in Change Stream listener: {e}. "
                "Switching to robust background polling (every 30 seconds)..."
            )
            self._run_polling()

    def _run_polling(self):
        """Fallback polling mechanism (runs every 30s) when Change Stream is not supported."""
        logger.info("Robust settings polling started (Interval: 30s).")
        while self.running:
            try:
                self._reload_and_update()
            except Exception as e:
                logger.error(f"[SettingsWatcher] Polling reload error: {e}")
            
            # Non-blocking sleep: check the self.running flag periodically
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(0.5)

    def _reload_and_update(self):
        """Reload configurations from MongoDB and safely apply camera URL updates."""
        old_url = CameraConfig.RTSP_URL
        # Reload from MongoDB
        success = CameraConfig.load_from_mongodb(mongo_db)
        if success:
            new_url = CameraConfig.RTSP_URL
            # If RTSP URL has changed, perform dynamic hot-reconnection
            if old_url != new_url:
                logger.info(
                    f"RTSP URL change detected!\n"
                    f" - Old: {self.camera._mask_url(str(old_url))}\n"
                    f" - New: {self.camera._mask_url(str(new_url))}"
                )
                # Disconnect the old camera stream
                self.camera.disconnect()
                # Connect to the new stream
                connected = self.camera.connect(new_url=new_url)
                if connected:
                    logger.success("Successfully hot-reconnected to the new RTSP URL!")
                else:
                    logger.error("Failed to hot-reconnect to the new RTSP URL.")
