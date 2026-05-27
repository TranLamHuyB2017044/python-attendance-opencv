import time
import threading
import queue
from pathlib import Path
from loguru import logger
from typing import Optional

from src.config import GoogleDriveConfig, PROJECT_ROOT


class GoogleDriveUploader:
    def __init__(self):
        self.enabled = GoogleDriveConfig.ENABLED
        self.use_service_account = GoogleDriveConfig.USE_SERVICE_ACCOUNT
        self.service_account_path = PROJECT_ROOT / GoogleDriveConfig.SERVICE_ACCOUNT_FILE
        self.credentials_path = PROJECT_ROOT / GoogleDriveConfig.CREDENTIALS_FILE
        self.token_path = PROJECT_ROOT / GoogleDriveConfig.TOKEN_FILE
        self.folder_id = GoogleDriveConfig.FOLDER_ID

        self.service = None
        self.upload_queue = queue.Queue()
        self.upload_thread = None
        self._running = False

        if self.enabled:
            self._init_service()
            self._start_upload_worker()

    def _init_service(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google.oauth2.service_account import Credentials as ServiceAccountCredentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            SCOPES = ['https://www.googleapis.com/auth/drive.file']

            creds = None

            if self.use_service_account:
                if not self.service_account_path.exists():
                    logger.warning(f"[GoogleDrive] Không tìm thấy service_account.json tại {self.service_account_path}")
                    logger.warning("[GoogleDrive] Vui lòng tạo Service Account và tải key JSON")
                    self.enabled = False
                    return
                creds = ServiceAccountCredentials.from_service_account_file(
                    str(self.service_account_path),
                    scopes=SCOPES
                )
                logger.info("[GoogleDrive] Sử dụng Service Account để xác thực")
            else:
                if self.token_path.exists():
                    creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        if not self.credentials_path.exists():
                            logger.warning(f"[GoogleDrive] Không tìm thấy credentials.json tại {self.credentials_path}")
                            logger.warning("[GoogleDrive] Vui lòng tạo credentials.json từ Google Cloud Console hoặc dùng Service Account")
                            self.enabled = False
                            return
                        flow = InstalledAppFlow.from_client_secrets_file(
                            str(self.credentials_path), SCOPES
                        )
                        creds = flow.run_local_server(port=0)
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                logger.info("[GoogleDrive] Sử dụng OAuth2 để xác thực")

            self.service = build('drive', 'v3', credentials=creds)
            logger.success("[GoogleDrive] Đã kết nối thành công!")
        except ImportError as e:
            logger.warning(f"[GoogleDrive] Thiếu thư viện Google Drive API: {e}")
            logger.warning("[GoogleDrive] Vui lòng cài đặt: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            self.enabled = False
        except Exception as e:
            logger.error(f"[GoogleDrive] Lỗi khởi tạo service: {e}")
            self.enabled = False

    def _ensure_folder_exists(self) -> Optional[str]:
        if not self.service:
            return None

        if self.folder_id:
            return self.folder_id

        try:
            query = "name='Attendance Daily Videos' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            items = results.get('files', [])

            if items:
                self.folder_id = items[0]['id']
                logger.info(f"[GoogleDrive] Đã tìm thấy folder: {self.folder_id}")
                return self.folder_id

            file_metadata = {
                'name': 'Attendance Daily Videos',
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            self.folder_id = folder.get('id')
            logger.success(f"[GoogleDrive] Đã tạo folder mới: {self.folder_id}")
            return self.folder_id

        except Exception as e:
            logger.error(f"[GoogleDrive] Lỗi tạo/tìm folder: {e}")
            return None

    def _upload_file(self, file_path: Path):
        if not self.service or not self.enabled:
            return False

        try:
            from googleapiclient.http import MediaFileUpload

            folder_id = self._ensure_folder_exists()
            if not folder_id:
                return False

            date_folder = file_path.parent.name
            file_name = file_path.name

            parent_folder_id = self._get_or_create_subfolder(folder_id, date_folder)
            if not parent_folder_id:
                parent_folder_id = folder_id

            file_metadata = {
                'name': file_name,
                'parents': [parent_folder_id]
            }

            media = MediaFileUpload(
                str(file_path),
                mimetype='video/mp4',
                resumable=True
            )

            logger.info(f"[GoogleDrive] Đang upload: {file_name}...")
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            logger.success(f"[GoogleDrive] Upload thành công: {file_name} (ID: {file.get('id')})")
            return True

        except Exception as e:
            logger.error(f"[GoogleDrive] Lỗi upload {file_path.name}: {e}")
            return False

    def _get_or_create_subfolder(self, parent_id: str, folder_name: str) -> Optional[str]:
        try:
            query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            items = results.get('files', [])

            if items:
                return items[0]['id']

            file_metadata = {
                'name': folder_name,
                'parents': [parent_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            return folder.get('id')

        except Exception as e:
            logger.warning(f"[GoogleDrive] Lỗi tạo subfolder {folder_name}: {e}")
            return None

    def _start_upload_worker(self):
        self._running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        logger.info("[GoogleDrive] Upload worker đã khởi động")

    def _upload_worker(self):
        while self._running:
            try:
                file_path = self.upload_queue.get()
                if file_path is None:
                    break

                self._upload_file(file_path)
                self.upload_queue.task_done()
                time.sleep(1)

            except Exception as e:
                logger.error(f"[GoogleDrive] Upload worker error: {e}")
                time.sleep(5)

    def queue_upload(self, file_path: Path):
        if not self.enabled:
            return

        if not file_path.exists():
            logger.warning(f"[GoogleDrive] File không tồn tại: {file_path}")
            return

        self.upload_queue.put(file_path)
        logger.info(f"[GoogleDrive] Đã thêm vào hàng đợi upload: {file_path.name}")

    def shutdown(self):
        self._running = False
        if self.upload_queue:
            self.upload_queue.put(None)
        if self.upload_thread and self.upload_thread.is_alive():
            self.upload_thread.join(timeout=10)
        logger.info("[GoogleDrive] Đã tắt upload worker")
