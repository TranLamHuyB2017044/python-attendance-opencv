import time
import threading
from loguru import logger

from src.config import CameraConfig, MongoDbConfig
from src.attendance.mongodb_mgr import mongo_db

class SettingsWatcher:
    """Poll the attendance API for effective camera settings every 10 seconds."""

    def __init__(self, camera_instance):
        self.camera = camera_instance
        self.running = False
        self.thread = None
        self.last_settings = None

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
        while self.running:
            try:
                settings = mongo_db.get_settings(CameraConfig.WATCHED_SETTING_KEYS, MongoDbConfig.COMPANY_ID)
                if settings != self.last_settings:
                    logger.info("[SettingsWatcher] Phát hiện thay đổi setting, reload config...")
                    if self._reload_and_update(settings):
                        self.last_settings = settings
            except Exception as e:
                logger.error(f"[SettingsWatcher] Polling reload error: {e}")
            
            # Non-blocking sleep: check the self.running flag periodically
            for _ in range(20):
                if not self.running:
                    break
                time.sleep(0.5)

    def _reload_and_update(self, settings):
        """Reload configurations through the attendance API and safely apply camera URL updates."""
        old_url = CameraConfig.RTSP_URL
        success = CameraConfig.load_from_mongodb(mongo_db, settings=settings)
        if success and CameraConfig.RTSP_URL != old_url:
            new_url = CameraConfig.RTSP_URL
            logger.info(f"[SettingsWatcher] Config loaded - Old URL: {self.camera._mask_url(str(old_url))} | New URL: {self.camera._mask_url(str(new_url))}")
            logger.info("[SettingsWatcher] Đang ngắt kết nối camera cũ...")
            self.camera.disconnect()
            logger.info("[SettingsWatcher] Đang kết nối lại camera với config mới...")
            connected = self.camera.connect(new_url=new_url)
            if connected:
                logger.success("Successfully hot-reconnected to the RTSP URL!")
            else:
                logger.error("Failed to hot-reconnect to the RTSP URL.")
        return success
