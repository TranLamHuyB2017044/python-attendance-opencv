import time
import cv2
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Any, Optional
from collections import deque

from src.config import DailyVideoConfig, DAILY_VIDEOS_DIR, CameraConfig
from src.utils.time_manager import time_mgr
from src.recognition.google_drive_uploader import GoogleDriveUploader


class DailyVideoRecorder:
    def __init__(self):
        self.enabled = DailyVideoConfig.ENABLED
        self.idle_timeout = DailyVideoConfig.IDLE_TIMEOUT_SECONDS
        self.max_log_lines = DailyVideoConfig.MAX_LOG_LINES
        self.retention_days = DailyVideoConfig.RETENTION_DAYS

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_video_path: Optional[Path] = None
        self.session_start_time: Optional[float] = None
        self.last_activity_time: Optional[float] = None
        self.current_date: Optional[str] = None

        self.session_logs = deque(maxlen=100)
        self.webhook_logs = deque(maxlen=50)

        self._is_recording = False
        self.drive_uploader = GoogleDriveUploader()

        if self.enabled:
            logger.info("[DailyVideoRecorder] Đã khởi tạo, sẵn sàng ghi video hàng ngày")
            self._cleanup_old_videos()
        else:
            logger.info("[DailyVideoRecorder] Chức năng bị tắt")

    def log_webhook(self, user_id: str, user_name: str, status: str, is_unknown: bool = False):
        vn_now = time_mgr.get_accurate_time()
        time_str = vn_now.strftime("%H:%M:%S")
        log_entry = f"[{time_str}] Webhook: {user_name} ({user_id}) - {status}"
        self.webhook_logs.append(log_entry)
        self.session_logs.append(log_entry)
        logger.debug(f"[DailyVideoRecorder] {log_entry}")

    def log_activity(self, message: str):
        vn_now = time_mgr.get_accurate_time()
        time_str = vn_now.strftime("%H:%M:%S")
        log_entry = f"[{time_str}] {message}"
        self.session_logs.append(log_entry)

    def _get_current_date_str(self) -> str:
        vn_now = time_mgr.get_accurate_time()
        return vn_now.strftime("%Y-%m-%d")

    def _get_video_filename(self) -> str:
        vn_now = time_mgr.get_accurate_time()
        start_str = vn_now.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{start_str}_session.mp4"

    def _ensure_date_folder(self) -> Path:
        date_str = self._get_current_date_str()
        date_folder = DAILY_VIDEOS_DIR / date_str
        date_folder.mkdir(parents=True, exist_ok=True)
        return date_folder

    def start_session(self, frame_width: int, frame_height: int):
        if not self.enabled:
            return

        if self._is_recording:
            return

        try:
            date_folder = self._ensure_date_folder()
            video_filename = self._get_video_filename()
            self.current_video_path = date_folder / video_filename

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = CameraConfig.FPS
            self.video_writer = cv2.VideoWriter(
                str(self.current_video_path),
                fourcc,
                fps,
                (frame_width, frame_height)
            )

            self.session_start_time = time.time()
            self.last_activity_time = time.time()
            self.current_date = self._get_current_date_str()
            self._is_recording = True

            self.log_activity("Bắt đầu phiên ghi video")
            logger.success(f"[DailyVideoRecorder] Bắt đầu ghi: {self.current_video_path}")

        except Exception as e:
            logger.error(f"[DailyVideoRecorder] Lỗi khi bắt đầu phiên: {e}")
            self._cleanup()

    def stop_session(self):
        if not self._is_recording:
            return

        try:
            self.log_activity("Kết thúc phiên ghi video")

            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            if self.current_video_path:
                duration = time.time() - self.session_start_time
                logger.success(f"[DailyVideoRecorder] Đã lưu video: {self.current_video_path} (Thời lượng: {duration:.1f}s)")
                
                self.drive_uploader.queue_upload(self.current_video_path)

        except Exception as e:
            logger.error(f"[DailyVideoRecorder] Lỗi khi dừng phiên: {e}")
        finally:
            self._cleanup()

    def _cleanup_old_videos(self):
        try:
            if not DAILY_VIDEOS_DIR.exists():
                return

            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            deleted_count = 0
            deleted_size = 0

            for date_folder in DAILY_VIDEOS_DIR.iterdir():
                if not date_folder.is_dir():
                    continue

                try:
                    folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                    if folder_date < cutoff_date:
                        folder_size = sum(f.stat().st_size for f in date_folder.rglob('*') if f.is_file())
                        shutil.rmtree(date_folder)
                        deleted_count += 1
                        deleted_size += folder_size
                        logger.info(f"[DailyVideoRecorder] Đã xóa thư mục cũ: {date_folder.name} ({folder_size / (1024*1024):.2f} MB)")
                except ValueError:
                    continue

            if deleted_count > 0:
                logger.success(f"[DailyVideoRecorder] Đã dọn dẹp {deleted_count} thư mục cũ, tổng dung lượng: {deleted_size / (1024*1024):.2f} MB")

        except Exception as e:
            logger.error(f"[DailyVideoRecorder] Lỗi cleanup video cũ: {e}")

    def _cleanup(self):
        if self.video_writer:
            try:
                self.video_writer.release()
            except:
                pass
            self.video_writer = None

        self.current_video_path = None
        self.session_start_time = None
        self.last_activity_time = None
        self._is_recording = False

    def check_idle(self) -> bool:
        if not self._is_recording:
            return False

        if time.time() - self.last_activity_time > self.idle_timeout:
            self.stop_session()
            return True

        return False

    def _draw_overlay(self, frame: cv2.Mat, detected_faces: List[Any]) -> cv2.Mat:
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        vn_now = time_mgr.get_accurate_time()
        time_str = vn_now.strftime("%Y-%m-%d %H:%M:%S")

        cv2.rectangle(overlay, (10, 10), (350, 45), (0, 0, 0), -1)
        cv2.putText(overlay, time_str, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self._is_recording:
            elapsed = int(time.time() - self.session_start_time)
            status_text = f"REC {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
            cv2.rectangle(overlay, (w - 180, 10), (w - 10, 45), (0, 0, 255), -1)
            cv2.putText(overlay, status_text, (w - 170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(overlay, (w - 200, 28), 8, (0, 0, 255), -1)

        if detected_faces:
            y_offset = 60
            cv2.rectangle(overlay, (10, y_offset - 25), (300, y_offset + 25 * len(detected_faces)), (0, 0, 0, 180), -1)
            cv2.putText(overlay, "Nguoi trong khung:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20

            for face in detected_faces:
                face_name = getattr(face, 'name', 'Unknown')
                track_id = getattr(face, 'track_id', 'N/A')
                text = f"  ID:{track_id} - {face_name}"
                cv2.putText(overlay, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += 18

        log_y = h - 20
        recent_logs = list(self.session_logs)[-self.max_log_lines:]
        for log in reversed(recent_logs):
            cv2.rectangle(overlay, (10, log_y - 18), (w - 10, log_y + 5), (0, 0, 0, 160), -1)
            cv2.putText(overlay, log, (20, log_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
            log_y -= 22

        alpha = 0.6
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def write_frame(self, frame: cv2.Mat, detected_faces: List[Any]):
        if not self.enabled:
            return

        if frame is None:
            return

        h, w = frame.shape[:2]

        if not self._is_recording:
            self.start_session(w, h)

        if self._is_recording:
            current_date = self._get_current_date_str()
            if current_date != self.current_date:
                self.stop_session()
                self.start_session(w, h)

            self.last_activity_time = time.time()

            frame_with_overlay = self._draw_overlay(frame, detected_faces)

            if self.video_writer:
                self.video_writer.write(frame_with_overlay)
