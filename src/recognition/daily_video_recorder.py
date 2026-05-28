import time
import cv2
import shutil
import threading
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

        self.session_frames: List[Any] = []
        self.current_video_path: Optional[Path] = None
        self.session_start_time: Optional[float] = None
        self.last_activity_time: Optional[float] = None  # Thời gian có hoạt động thực tế (face detect/webhook)
        self.current_date: Optional[str] = None

        self.session_logs = deque(maxlen=100)
        self.webhook_logs = deque(maxlen=50)

        self._is_recording = False
        self._save_threads = []  # Theo dõi các thread lưu video
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
        self.last_activity_time = time.time()  # Cập nhật thời gian hoạt động
        logger.debug(f"[DailyVideoRecorder] {log_entry}")

    def log_activity(self, message: str):
        vn_now = time_mgr.get_accurate_time()
        time_str = vn_now.strftime("%H:%M:%S")
        log_entry = f"[{time_str}] {message}"
        self.session_logs.append(log_entry)
        self.last_activity_time = time.time()  # Cập nhật thời gian hoạt động

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

    def _draw_overlay(self, frame: cv2.Mat, detected_faces: List[Any]) -> cv2.Mat:
        render = frame.copy()
        h, w = render.shape[:2]

        vn_now = time_mgr.get_accurate_time()
        time_str = vn_now.strftime("%Y-%m-%d %H:%M:%S")

        cv2.rectangle(render, (10, 10), (380, 50), (0, 0, 0), -1)
        cv2.putText(render, time_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if self._is_recording and self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            status_text = f"REC {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
            cv2.rectangle(render, (w - 200, 10), (w - 10, 50), (0, 0, 255), -1)
            cv2.putText(render, status_text, (w - 190, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(render, (w - 220, 30), 10, (0, 0, 255), -1)

        if detected_faces:
            y_offset = 70
            box_height = 30 + 25 * len(detected_faces)
            cv2.rectangle(render, (10, y_offset - 30), (350, y_offset + box_height), (0, 0, 0, 200), -1)
            cv2.putText(render, "=== Nguoi trong khung ===", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            y_offset += 25

            for face in detected_faces:
                face_name = getattr(face, 'name', 'Unknown')
                track_id = getattr(face, 'track_id', 'N/A')
                text = f"  ID:{track_id} - {face_name}"
                cv2.putText(render, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 2)
                y_offset += 22

        log_panel_width = 600
        log_panel_x = w - log_panel_width - 10
        recent_logs = list(self.session_logs)[-self.max_log_lines:]
        
        if recent_logs:
            panel_height = 30 + 22 * len(recent_logs)
            log_y = h - 20
            
            cv2.rectangle(render, (log_panel_x, log_y - panel_height + 10), (w - 10, log_y + 10), (0, 0, 0, 220), -1)
            cv2.putText(render, "=== LOG PHIEN ===", (log_panel_x + 20, log_y - panel_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            
            current_y = log_y - panel_height + 55
            for log in recent_logs:
                cv2.putText(render, log, (log_panel_x + 20, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 2)
                current_y += 22

        return render

    def _save_video_async(self, path, frames, fps):
        try:
            if not frames or len(frames) == 0:
                logger.warning("[DailyVideoRecorder] Không có frame để lưu")
                return

            h, w = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, (int(w), int(h)))
            
            if not out.isOpened():
                logger.warning(f"[DailyVideoRecorder] Không thể mở mp4v, thử dùng XVID...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                path_avi = path.replace('.mp4', '.avi')
                path = path_avi
                out = cv2.VideoWriter(path_avi, fourcc, fps, (int(w), int(h)))
            
            if out.isOpened():
                for vf in frames:
                    out.write(vf)
                out.release()
                file_size = Path(path).stat().st_size
                logger.success(f"[DailyVideoRecorder] Đã lưu video: {path} ({file_size / 1024:.1f} KB)")
                
                self.drive_uploader.queue_upload(Path(path))
            else:
                logger.error(f"[DailyVideoRecorder] Không thể lưu video!")
        except Exception as e:
            logger.error(f"[DailyVideoRecorder] Lỗi lưu video: {e}")
        finally:
            # Loại bỏ thread khỏi danh sách sau khi hoàn thành
            if hasattr(threading.current_thread(), '_thread_id'):
                self._save_threads = [t for t in self._save_threads if t.is_alive()]

    def start_session(self, frame_width: int, frame_height: int):
        if not self.enabled:
            return

        if self._is_recording:
            return

        try:
            date_folder = self._ensure_date_folder()
            video_filename = self._get_video_filename()
            self.current_video_path = date_folder / video_filename

            self.session_frames = []
            self.session_start_time = time.time()
            self.last_activity_time = time.time()
            self.last_save_time = time.time()
            self.current_date = self._get_current_date_str()
            self._is_recording = True

            self.log_activity("Bắt đầu phiên ghi video")
            logger.success(f"[DailyVideoRecorder] Bắt đầu ghi: {self.current_video_path}")

        except Exception as e:
            logger.error(f"[DailyVideoRecorder] Lỗi khi bắt đầu phiên: {e}")
            self._cleanup()

    def stop_session(self, save_immediately: bool = False):
        if not self._is_recording:
            return

        try:
            self.log_activity("Kết thúc phiên ghi video")

            if self.current_video_path and len(self.session_frames) >= 1:
                duration = time.time() - self.session_start_time
                actual_fps = len(self.session_frames) / duration if duration > 0 else 15.0
                actual_fps = max(5.0, min(30.0, actual_fps))
                
                logger.info(f"[DailyVideoRecorder] Đang lưu {len(self.session_frames)} frames...")
                t = threading.Thread(
                    target=self._save_video_async, 
                    args=(str(self.current_video_path), self.session_frames, actual_fps), 
                    daemon=False  # Không dùng daemon để đảm bảo lưu xong
                )
                self._save_threads.append(t)
                t.start()
                
                if save_immediately:
                    t.join()  # Đợi lưu xong nếu cần
            else:
                logger.warning(f"[DailyVideoRecorder] Không đủ frame để lưu (chỉ có {len(self.session_frames)} frame)")

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
        self.session_frames = []
        self.current_video_path = None
        self.session_start_time = None
        self.last_activity_time = None
        self._is_recording = False

    def check_idle(self) -> bool:
        if not self._is_recording:
            return False

        # Kiểm tra thay đổi ngày trước (độc lập với idle)
        current_date = self._get_current_date_str()
        if current_date != self.current_date:
            logger.info(f"[DailyVideoRecorder] Đổi ngày (từ {self.current_date} sang {current_date}), dừng phiên cũ và lưu video")
            self.stop_session()
            return True

        # Kiểm tra idle timeout
        if time.time() - self.last_activity_time > self.idle_timeout:
            logger.info(f"[DailyVideoRecorder] Phát hiện idle (không hoạt động {self.idle_timeout}s), dừng phiên và lưu video")
            self.stop_session()
            return True

        return False

    def force_save_and_stop(self):
        """Buộc lưu video và dừng phiên (gọi khi app tắt)"""
        if self._is_recording:
            logger.info("[DailyVideoRecorder] Buộc lưu video (app đang tắt)...")
            self.stop_session(save_immediately=True)
        
        # Đợi tất cả các thread lưu video hoàn thành
        logger.info("[DailyVideoRecorder] Đợi các thread lưu video hoàn thành...")
        for t in self._save_threads:
            if t.is_alive():
                t.join(timeout=30)
        logger.success("[DailyVideoRecorder] Đã hoàn thành tất cả lưu trữ")

    def write_frame(self, frame: cv2.Mat, detected_faces: List[Any]):
        if not self.enabled:
            logger.debug("[DailyVideoRecorder] Chức năng bị tắt, không ghi frame")
            return

        if frame is None:
            logger.debug("[DailyVideoRecorder] Frame là None, bỏ qua")
            return

        h, w = frame.shape[:2]

        if not self._is_recording:
            logger.debug("[DailyVideoRecorder] Chưa ghi, gọi start_session()")
            self.start_session(w, h)

        if self._is_recording:
            # Kiểm tra thay đổi ngày (đã có trong check_idle(), nhưng kiểm tra thêm ở đây để đảm bảo)
            current_date = self._get_current_date_str()
            if current_date != self.current_date:
                logger.info(f"[DailyVideoRecorder] Đổi ngày, dừng phiên cũ, bắt đầu phiên mới")
                self.stop_session()
                self.start_session(w, h)
                return

            # Chỉ cập nhật last_activity_time nếu có phát hiện khuôn mặt (hoạt động thực tế)
            if detected_faces and len(detected_faces) > 0:
                self.last_activity_time = time.time()

            frame_with_overlay = self._draw_overlay(frame, detected_faces)

            if frame_with_overlay is not None:
                self.session_frames.append(frame_with_overlay)
                if len(self.session_frames) % 30 == 0:
                    logger.debug(f"[DailyVideoRecorder] Đã có {len(self.session_frames)} frames trong buffer")
