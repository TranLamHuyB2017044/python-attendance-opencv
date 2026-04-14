import time
import numpy as np
from loguru import logger
from typing import List, Dict, Any
import cv2
import requests
import threading
from src.config import RecognitionConfig, CAPTURES_DIR, MODELS_DIR, ApiConfig, CameraConfig, WebhookConfig
from src.attendance.mongodb_mgr import mongo_db
from src.utils.string_utils import remove_accents
from src.utils.time_manager import time_mgr
from src.recognition.async_spoof import AsyncSpoofChecker

class FaceTracker:
    """
    Tracks faces across frames to ensure stability before recognition.
    Includes a cooldown mechanism and unknown-face attempt logic with audio.
    """
    def __init__(self, threshold_seconds=0):
        self.active_faces: Dict[int, Dict[str, Any]] = {} # {id: data}
        self.face_id_counter = 0
        self.recent_snapshots = [] # History of successful snapshots for UI overlay
        self.threshold_seconds = threshold_seconds
        
        # Motion Tracking
        self.prev_small_gray = None
        
        # Global cooldowns: {user_id: last_detection_time}
        self.user_cooldowns: Dict[str, float] = {}
        
        # Webhook cooldown for known users: {user_id: last_webhook_time}
        # Prevents sending duplicate webhooks within 15 minutes
        self.webhook_sent_time: Dict[str, float] = {}
        
        # Track successful webhook counts per user session
        self.successful_webhook_counts: Dict[str, int] = {}
        
        # Unknown face webhook counter (max 10, reset on successful checkin)
        self.unknown_webhook_count = 0
        self.unknown_webhook_limit = 10

        # ─── Async Anti-Spoofing ────────────────────────────────────────────
        # Chạy MiniFASNet song song với GATHERING — không block recognition
        self.spoof_checker = AsyncSpoofChecker(
            model_dir=str(MODELS_DIR / "anti_spoof")
        ) if RecognitionConfig.ANTI_SPOOFING_ENABLED else None
        if self.spoof_checker:
            logger.info("[Tracker] Async Anti-Spoofing ENABLED (parallel mode)")
        else:
            logger.info("[Tracker] Anti-Spoofing DISABLED")

    def _send_telegram_alert(self, title, details, image=None):
        """Sends an enhanced Telegram alert with device and shift info."""
        if RecognitionConfig.TEST_MODE:
            return
        try:
            from src.utils.telegram_bot import send_telegram_report
            from src.services.ping_service import _build_devices_info
            from src.utils.time_manager import time_mgr
            
            # 1. Get current VN time and Shift
            vn_now = time_mgr.get_accurate_time()
            time_str = vn_now.strftime("%H:%M:%S")
            hour_val = vn_now.hour * 100 + vn_now.minute
            
            shift = "Ngoai gio"
            if 730 <= hour_val <= 1200:
                shift = "Ca Sang"
            elif 1250 <= hour_val <= 1800:
                shift = "Ca Chieu"
                
            # 2. Get Device Info
            dev_info = _build_devices_info()
            host_name = dev_info.get('host', {}).get('hostname', 'Unknown')
            cam_name = dev_info.get('camera', {}).get('name', 'Unknown Camera')
            
            # 3. Build Message
            # Note: send_telegram_report already handles formatting, we just provide the text
            message = (
                f"Thoi gian: {time_str} ({shift})\n"
                f"Thiet bi: {host_name}\n"
                f"Camera: {cam_name}\n"
                f"Chi tiet: {details}"
            )
            
            send_telegram_report(title, message, image=image)
        except Exception as e:
            logger.error(f"Error sending enhanced Telegram alert: {e}")

    def _send_user_webhook(self, user_id, user_name, status, is_unknown=False, custom_voice_text=None):
        """
        Sends user detection info to the configured webhook URL in a background thread.
        
        Deduplication rules:
        - Known users: Max 1 webhook per 15 minutes per user_id
        - Unknown faces: Max 10 webhooks total, reset when any known user checks in
        """
        # ── Always push to local log bus (for TEST_MODE log panel) ────────────────
        try:
            from src.utils.webhook_log_bus import push_sent
            import datetime
            push_sent(
                user_id=str(user_id),
                user_name=user_name,
                status=status or "DETECTED",
                voice_text=custom_voice_text or "",
                time_str=datetime.datetime.now().strftime("%H:%M:%S")
            )
        except Exception:
            pass


        import os
        current_time = time.time()  # Định nghĩa current_time local để tránh lỗi reference

        # === CROSS-PROCESS DEDUPLICATION LOGIC TO PREVENT MULTI-PROCESS RACE CONDITIONS ===
        if is_unknown:
            if self.unknown_webhook_count >= self.unknown_webhook_limit:
                return
            self.unknown_webhook_count += 1
        else:
            # OS Cross-Process File Lock (Data Dir Shared Between All Instances)
            from src.config import DATA_DIR

            safe_id = "".join(x for x in str(user_id) if x.isalnum())
            lock_file = DATA_DIR / f"webhook_lock_{safe_id}.txt"

            try:
                if lock_file.exists():
                    last_mtime = os.path.getmtime(lock_file)
                    elapsed = current_time - last_mtime
                    if status != 'COOLDOWN' and elapsed < 5:
                        return  # Block duplicate webhook trong 5 giây

                with open(lock_file, "w") as f:
                    f.write(str(current_time))
            except Exception:
                pass

            if self.unknown_webhook_count > 0:
                self.unknown_webhook_count = 0

        def thread_task():
            try:
                from src.utils.string_utils import remove_accents
                from src.utils.time_manager import time_mgr
                
                # === PREPARE PAYLOAD ===
                # 1. Format time HH:MM:SS from VN Time
                vn_now = time_mgr.get_accurate_time()
                time_str = vn_now.strftime("%H:%M:%S")
                
                # 2. Format user_name without accents for the JSON field
                name_no_accents = remove_accents(user_name)
                
                # 3. Format voice text for TTS
                name_parts = user_name.strip().split()
                # Lấy tên đệm và tên (2 phần cuối của chuỗi tên)
                short_name = " ".join(name_parts[-2:]) if len(name_parts) >= 2 else user_name

                if custom_voice_text:
                    voice_text = custom_voice_text
                elif status in ["IN", "OUT"]:
                    voice_text = f"{short_name} đã chấm công"
                elif status == "SPOOF":
                    voice_text = "Cảnh báo: Phát hiện hành vi giả mạo khuôn mặt"
                elif status == "COOLDOWN":
                    voice_text = f"{short_name} đã truy cập gần đây"
                else:
                    # Default for unknown/unauthorized
                    voice_text = "Vui lòng thử lại"

                payload = {
                    "user_id": user_id,
                    "user_name": name_no_accents,
                    "status": status or "DETECTED",
                    "voice_text": voice_text,
                    "time": time_str
                }

                response = requests.post(
                    WebhookConfig.USER_WEBHOOK_URL,
                    json=payload,
                    timeout=5
                )
                if response.status_code == 200:
                    self.successful_webhook_counts[user_id] = self.successful_webhook_counts.get(user_id, 0) + 1
                    count = self.successful_webhook_counts[user_id]
                    logger.success(f"=====================================================")
                    logger.success(f"🚀 WEBHOOK GỬI THÀNH CÔNG -> [ {user_name} ] - Status: {status}")
                    logger.success(f"📊 Đây là webhook thành công lần thứ {count} của nhân viên này.")
                    logger.success(f"=====================================================")
                else:
                    error_msg = f"WEBHOOK FAILED -> [ {user_name} ] - Code: {response.status_code}"
                    logger.warning(error_msg)
                    from src.utils.telegram_bot import send_telegram_report
                    send_telegram_report("Webhook Fail", f"Nhân viên: {user_name} ({user_id})\nStatus: {status}\nHTTP Code: {response.status_code}")
            except Exception as e:
                error_msg = f"Error sending webhook: {e}"
                logger.error(error_msg)
                from src.utils.telegram_bot import send_telegram_report
                send_telegram_report("Webhook Fail", f"Nhân viên: {user_name} ({user_id})\nStatus: {status}\nError: {str(e)}")

        # Run in background to not block the tracking loop
        threading.Thread(target=thread_task, daemon=True).start()

    def _get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def _auto_learn_face(self, frame, face, user_id, user_name, birthday, company_id, attendance_mgr):
        """Tự động học mẫu khuôn mặt mới từ ảnh chấm công"""
        # Auto-learn logic implementation
        def task():
            try:
                # 1. Trích xuất và nén ảnh khuôn mặt
                bbox = face.bbox.astype(int)
                h, w = frame.shape[:2]
                
                # Mở rộng bbox một chút để lấy ảnh đẹp hơn ( enrollment-style )
                pad_w = int((bbox[2] - bbox[0]) * 0.2)
                pad_h = int((bbox[3] - bbox[1]) * 0.2)
                
                x1 = max(0, bbox[0] - pad_w)
                y1 = max(0, bbox[1] - pad_h)
                x2 = min(w, bbox[2] + pad_w)
                y2 = min(h, bbox[3] + pad_h)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    return

                success, encoded_img = cv2.imencode('.webp', face_crop, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                if not success:
                    return
                
                image_blob = encoded_img.tobytes()
                
                # 2. Lưu vào MongoDB
                from src.attendance.mongodb_mgr import mongo_db
                image_id = mongo_db.save_enrollment_image(user_id, company_id, image_blob)
                if not image_id:
                    logger.warning(f"Auto-learn: Failed to save enrollment image for {user_id}")
                    return
                
                # 3. Lưu vào Qdrant với image_id
                ok = attendance_mgr.upsert_user(
                    user_name=user_name,
                    user_id=user_id,
                    birthday=birthday,
                    embeddings=[face.normed_embedding],
                    company_id=company_id,
                    enrollment_image_ids=[image_id]
                )
                
                if ok:
                    logger.info(f"Auto-learned new sample for {user_name} (ID: {user_id}), total points updated.")
                else:
                    logger.warning(f"Auto-learn: Failed to update Qdrant for {user_id}")
                    
            except Exception as e:
                logger.error(f"Error in _auto_learn_face: {e}")

        threading.Thread(target=task, daemon=True).start()

    def _save_log_with_bbox(self, frame, face, user_id, user_name, score, is_known=True, status=None, company_id=None, unknown_attempt=0, birthday="N/A", vector_count=0, f_data=None):
        """Sync state check + Async heavy saving."""
        if frame is None: return None, None
        
        from src.attendance.mongodb_mgr import mongo_db
        # 1. IMMEDIATE SYNC CHECK (Fast)
        final_status = status
        if is_known and status is None:
            final_status = mongo_db.get_attendance_status(user_id, company_id=company_id) 
            
        # --- BLOCK NEW SAVES IF IN COOLDOWN ---
        if final_status == 'COOLDOWN':
            return None, final_status
            
        # =========================================================================
        # --- CROSS-PROCESS HARD-LOCK FOR DB WRITES & WEBHOOKS ---
        # Ngăn chặn hoàn toàn việc ghi log DB đúp và bắn webhook đúp do chạy nền song song với UI
        # =========================================================================
        if is_known and final_status in ['IN', 'OUT']:
            from src.config import DATA_DIR
            import os
            import time
            
            safe_id = "".join(x for x in str(user_id) if x.isalnum())
            lock_file = DATA_DIR / f"attendance_lock_{safe_id}.txt"
            current_time = time.time()
            
            try:
                if lock_file.exists():
                    last_mtime = os.path.getmtime(lock_file)
                    elapsed = current_time - last_mtime
                    # Khoá chặt trong 5 giây cho mọi nỗ lực IN/OUT
                    if elapsed < 5:
                        logger.warning(f"[RACE CONDITION BLOCKED] Rejected duplicate {final_status} for {user_name} (locked {int(elapsed)}s ago).")
                        return None, 'COOLDOWN'
                
                # Chiếm quyền ghi (Acquire Lock)
                with open(lock_file, "w") as f:
                    f.write(str(current_time))
            except Exception:
                pass
        
        # 2. Visual Prep: SMART CROP (Maintain 16:9 Aspect Ratio centered on face)
        log_frame = frame.copy()
        bbox = face.bbox.astype(int)
        h_orig, w_orig = frame.shape[:2]
        
        # Center of face
        face_cx = (bbox[0] + bbox[2]) // 2
        face_cy = (bbox[1] + bbox[3]) // 2
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]

        # Target crop size (approx 4x face width for good context)
        # We need width = 1.77 * height (16:9)
        crop_h = int(face_h * 4.0)
        crop_w = int(crop_h * (16/9))
        
        # Ensure crop isn't too small or too large
        crop_h = min(h_orig, max(200, crop_h))
        crop_w = int(crop_h * (16/9))
        
        if crop_w > w_orig:
            crop_w = w_orig
            crop_h = int(crop_w * (9/16))

        # Calculate coordinates centered on face
        x1 = max(0, face_cx - crop_w // 2)
        y1 = max(0, face_cy - crop_h // 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Shift crop if it hits right/bottom edge
        if x2 > w_orig:
            x2 = w_orig
            x1 = max(0, x2 - crop_w)
        if y2 > h_orig:
            y2 = h_orig
            y1 = max(0, y2 - crop_h)
            
        # Draw small marker on the original before cropping
        color = (0, 255, 0) if is_known else (0, 255, 255)
        cv2.rectangle(log_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Calculate relative coordinates for labels in the crop
        rel_x = bbox[0] - x1
        rel_y = bbox[1] - y1
        rel_x2 = bbox[2] - x1
        rel_y2 = bbox[3] - y1
        
        # Crop the face area
        save_frame = log_frame[y1:y2, x1:x2]
        
        def async_save_task():
            try:
                from src.config import RecognitionConfig, CAPTURES_DIR, ApiConfig, CameraConfig
                from src.utils.string_utils import remove_accents
                
                # Draw Labels exactly as detection frame
                ts, _ = time_mgr.get_formatted_time()
                current_time_str = ts.split(' ')[-1] # Only HH:MM:SS
                
                # Main Name Label
                main_label = f"{remove_accents(user_name)} ({score:.2f})"
                if not is_known and unknown_attempt > 0:
                    main_label = f"Nguoi la #{unknown_attempt}"
                
                # Draw Name Label above box
                cv2.putText(save_frame, main_label, (rel_x, rel_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                
                # Camera & Time Label
                cam_time_label = f"{CameraConfig.CAMERA_NAME} | {current_time_str}"
                cv2.putText(save_frame, cam_time_label, (rel_x, rel_y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                if is_known:
                    meta_info = [
                        f"ID: {user_id}",
                        f"Gio: {current_time_str}",
                        f"Mau: {vector_count}"
                    ]
                    
                    for i, text in enumerate(meta_info):
                        pos = (rel_x, rel_y2 + 25 + (i * 22))
                        cv2.putText(save_frame, text, pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                
                # --- OPTIMIZATION: Resize to 640x360 and compress quality ---
                # Dùng INTER_CUBIC để ảnh sắc nét hơn và thêm bộ lọc sắc nét nhẹ
                final_img = cv2.resize(save_frame, (640, 360), interpolation=cv2.INTER_CUBIC)
                
                # Nâng nhẹ độ tương phản và độ nét
                sharpen_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
                final_img = cv2.filter2D(final_img, -1, sharpen_kernel)

                raw_company_name = mongo_db.get_company_name(company_id) if company_id else "unknown_company"
                company_folder = remove_accents(raw_company_name).replace(" ", "_")
                
                # Tạo thư mục con theo tên người (thư mục tên)
                if is_known:
                    user_folder = f"{remove_accents(user_name).replace(' ','_')}_{user_id}"
                    target_dir = CAPTURES_DIR / company_folder / user_folder
                else:
                    target_dir = CAPTURES_DIR / company_folder / "Nguoi_La"
                
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # --- THIẾT LẬP VIDEO ĐỐI SOÁT ---
                video_path = None
                if f_data is not None:
                    if f_data.get('video_path') is None:
                        v_dir = target_dir.parent / "videos"
                        v_dir.mkdir(parents=True, exist_ok=True)
                        v_path = v_dir / f"{int(time.time())}_{remove_accents(user_name).replace(' ','_')}.mp4"
                        f_data['video_path'] = str(v_path)
                    video_path = f_data['video_path']

                # Tên ảnh chỉ cần timestamp vì đã nằm trong thư mục tên
                img_name = f"{int(time.time())}.webp"
                img_path = str(target_dir / img_name)
                
                # Lưu ảnh với chất lượng WebP 75 (Rõ nét hơn nhưng vẫn tối ưu dung lượng)
                cv2.imwrite(img_path, final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 75])
                
                # LƯU ẢNH OVERLAY (Tối đa 3 ảnh hiển thị trên UI)
                if is_known and final_status in ['IN', 'OUT']:
                    self.recent_snapshots.append({
                        "img": final_img.copy(),
                        "name": remove_accents(user_name),
                        "time": time.time(),
                        "status": final_status
                    })
                    if len(self.recent_snapshots) > 3:
                        self.recent_snapshots.pop(0)

                # Actual DB Commit
                mongo_db.log_attendance(user_id, user_name, status=final_status, frame=final_img, company_id=company_id, unknown_attempt=unknown_attempt, video_path=video_path)
            except Exception as e:
                logger.error(f"Async log error: {e}")

        # Start background saving
        threading.Thread(target=async_save_task, daemon=True).start()
        
        # Return the actual REAL status so Voice says it right
        return "processing_url", final_status

    def update(self, detected_faces, attendance_mgr, face_rec=None, frame=None, company_id=None, only_recognize=False):
        """
        Assigns IDs and decides when to trigger recognition or alerts.
        """
        current_time = time.time()
        updated_faces_map = {}
        
        # Use provided company_id or fallback
        active_company = company_id
        
        # 1. QUÉT CHUYỂN ĐỘNG TOÀN KHUNG HÌNH VÀ VÙNG THÂN NGƯỜI (MOTION DETECTION)
        # Ý tưởng: "Từ góc bìa đến khung detect" - Cắt toàn bộ nền để xem có sự di chuyển không.
        motion_pixels = 0
        is_frame_static = False
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Thu nhỏ tối đa để tối ưu CPU và loại bỏ nhiễu hạt (noise) của camera
            small_gray = cv2.resize(gray, (160, 120)) 
            
            if self.prev_small_gray is not None:
                # Trừ 2 khung hình để tìm sự khác biệt pixel (Những chỗ có thay đổi sẽ biến thành màu trắng)
                diff = cv2.absdiff(self.prev_small_gray, small_gray)
                _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                motion_pixels = np.count_nonzero(diff_thresh)
                
                # Diện tích khung hình là 160*120 = 19200 pixels.
                # Nếu dưới 50 pixels thay đổi => Ảnh đóng băng tĩnh hoàn toàn (Ảnh giấy, điện thoại gắn giá đỡ cầm tay cứng)
                if motion_pixels < 50:
                    is_frame_static = True
                    
            self.prev_small_gray = small_gray
        
        # Keep track of IDs used in THIS frame to avoid double matching
        used_ids_in_frame = set()

        # Sequential Processing: Only allow 1 recognition attempt per frame to save CPU
        recognition_done_this_frame = False

        def calc_iou(b1, b2):
            x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / float(b1_area + b2_area - inter + 1e-6)

        for face in detected_faces:
            center = self._get_center(face.bbox)
            matched_id = None
            is_real = getattr(face, 'is_real', True)
            as_label = getattr(face, 'as_label', 1) 
            
            # Sort active faces by combined IoU and Center Distance
            potential_matches = []
            face_w = face.bbox[2] - face.bbox[0]
            face_h = face.bbox[3] - face.bbox[1]
            max_dist = max(250, max(face_w, face_h) * 1.5)

            for f_id, f_data in self.active_faces.items():
                if f_id in used_ids_in_frame: continue
                prev_center = f_data['center']
                prev_bbox = f_data.get('bbox', [0, 0, 0, 0])
                
                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                iou = calc_iou(face.bbox, prev_bbox)
                
                if dist < max_dist or iou > 0.15: 
                    score = dist - iou * 1000 # High IoU rewards match heavily
                    potential_matches.append((score, f_id))
            
            if potential_matches:
                potential_matches.sort(key=lambda x: x[0])
                matched_id = potential_matches[0][1]
                used_ids_in_frame.add(matched_id)
            
            if matched_id is None:
                matched_id = self.face_id_counter
                self.face_id_counter += 1
                f_data = {
                    'start_time':         current_time,
                    'last_seen':          current_time,
                    'center':             center,
                    'bbox':               face.bbox,
                    'status':             'GATHERING',
                    'user_data':          None,
                    'cooldown_remaining': 0,
                    'unknown_attempts':   0,
                    'last_attempt_time':  0,
                    'gathering_embeddings': [],
                    'gather_count':         0,
                    'video_frames':       [cv2.resize(frame, (640, 360))] if frame is not None else [],
                    'video_path':         None
                }
            else:
                f_data = self.active_faces[matched_id]
                f_data['last_seen'] = current_time
                f_data['center'] = center
                f_data['bbox'] = face.bbox
                if frame is not None and len(f_data.setdefault('video_frames', [])) < 900:
                    f_data['video_frames'].append(cv2.resize(frame, (640, 360)))

            # SYNC BACK: If we already verified this face in previous frames
            if f_data.get('liveness_verified'):
                face.as_label = 1 if f_data['status'] != 'SPOOF_DETECTED' else 0
                face.is_real = (face.as_label == 1)
                if f_data['user_data']:
                    face.name = remove_accents(f_data['user_data'].get('name', 'Unknown'))
                    face.as_score = f_data['user_data'].get('as_score', 1.0)

            # ———————————————————————————
            # RECOGNITION ATTEMPT LOGIC
            # ———————————————————————————
            wait_time = current_time - f_data.get('last_attempt_time', 0)
            can_attempt = False
            
            if f_data['status'] in ['RECOGNIZED', 'COOLDOWN']:
                can_attempt = False
            elif f_data['status'] == 'GATHERING':
                can_attempt = True
            elif f_data['status'] in ['RETRY_WAIT', 'UNAUTHORIZED', 'SPOOF_DETECTED']:
                # Xóa độ trễ (delay) để retry ngay lập tức (Real-time quét)
                f_data['status'] = 'GATHERING'
                f_data['gathering_embeddings'] = [] 
                f_data['gather_count'] = 0
                can_attempt = True

            if can_attempt and face_rec is not None and attendance_mgr is not None:
                GATHER_FRAMES = getattr(RecognitionConfig, 'GATHER_FRAMES', 1)

                # Bộ lọc Face Quality
                is_good_frame = True
                
                # 1. Box Bounds & Blur
                int_bbox = face.bbox.astype(int)
                x1, y1 = max(0, int_bbox[0]), max(0, int_bbox[1])
                x2, y2 = min(frame.shape[1], int_bbox[2]), min(frame.shape[0], int_bbox[3])
                crop_h, crop_w = y2 - y1, x2 - x1
                # Skip nếu mặt quá sát biên hoặc quá nhỏ
                if crop_h > 40 and crop_w > 40:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        blur_val = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                        if blur_val < 30: is_good_frame = False
                
                # 2. Angle (Pitch/Yaw/Roll)
                if hasattr(face, 'pose') and face.pose is not None:
                    pitch, yaw, roll = face.pose
                    if abs(yaw) > 30 or abs(pitch) > 30: is_good_frame = False
                
                if is_good_frame:
                    # Step 1: Extract embedding
                    face_rec.rec_model.get(frame, face)
                    emb = face.normed_embedding.copy()
                    
                    # Step 2: Accumulate
                    f_data.setdefault('gathering_embeddings', []).append(emb)
                    f_data['gather_count'] = len(f_data['gathering_embeddings'])
                else:
                    face.name = "Anh mo hoac goc nghieng..."
                    f_data['gather_count'] = len(f_data.setdefault('gathering_embeddings', []))

                # Step 3: Check if reached count
                if f_data['gather_count'] >= GATHER_FRAMES:
                    # Step 4: Recognition
                    embs = np.array(f_data['gathering_embeddings'])
                    avg_emb = embs.mean(axis=0)
                    norm = np.linalg.norm(avg_emb)
                    if norm > 0: avg_emb = avg_emb / norm

                    logger.debug(f"[Recognition] ID={matched_id}: Processing with {GATHER_FRAMES} frames")
                    vote = attendance_mgr.recognize(avg_emb)
                    
                    # Reset accumulator
                    f_data['gathering_embeddings'] = []
                    f_data['gather_count'] = 0

                    # Conclusion
                    is_known = vote.get('name', 'Unknown') != 'Unknown'

                    if is_known:
                        # Synchronous Spoof Check
                        if self.spoof_checker and frame is not None:
                            spoof_real, spoof_score = self.spoof_checker.check_sync(frame, face, timeout=0.15)
                            if not spoof_real:
                                logger.warning(f"[Spoof] ❌ SPOOF DETECTED — ID={matched_id} score={spoof_score:.2f}")
                                f_data['status'] = 'SPOOF_DETECTED'
                                f_data['last_attempt_time'] = current_time
                                
                                # Cơ chế chống Spam Spoof (5s báo 1 lần trên cùng 1 người)
                                last_spoof = f_data.get('last_spoof_alert', 0)
                                if current_time - last_spoof > 5.0:
                                    f_data['last_spoof_alert'] = current_time
                                    self._send_user_webhook("Spoof", "Ke gia mao", "SPOOF", is_unknown=True)
                                    self._send_telegram_alert(
                                        "CANH BAO GIA MAO", 
                                        f"Phat hien hanh vi gia mao khuon mat!\nID: {matched_id}\nScore: {spoof_score:.2f}",
                                        image=frame
                                    )
                                updated_faces_map[matched_id] = f_data
                                continue

                        # Correct realization
                        user_data = vote
                        user_id   = user_data.get('user_id', 'Unknown')
                        user_name = user_data.get('name', 'Unknown')
                        logger.info(f"[Recognition] ✅ {user_name} (score={vote.get('score',0):.3f})")

                        if only_recognize:
                            f_data['status'] = 'RECOGNIZED'
                            f_data['user_data'] = user_data
                        else:
                            target_cid = user_data.get('company_id') or active_company
                            
                            # Check session-based cooldown for UI
                            in_cooldown = False
                            if not RecognitionConfig.TEST_MODE and user_id in self.user_cooldowns:
                                elapsed = current_time - self.user_cooldowns[user_id]
                                if elapsed < RecognitionConfig.COOLDOWN_SECONDS and RecognitionConfig.COOLDOWN_SECONDS > 0:
                                    in_cooldown = True

                            if in_cooldown:
                                f_data['status'] = 'COOLDOWN'
                                f_data['user_data'] = user_data
                                f_data['cooldown_remaining'] = int(RecognitionConfig.COOLDOWN_SECONDS - (current_time - self.user_cooldowns[user_id]))
                                self._send_user_webhook(user_id, user_name, "COOLDOWN", is_unknown=False)
                            else:
                                f_data['status'] = 'RECOGNIZED'
                                f_data['user_data'] = user_data
                                self.user_cooldowns[user_id] = current_time
                                _, status = self._save_log_with_bbox(frame, face, user_id, user_name, user_data.get('score', 0.0),
                                                                       company_id=target_cid, birthday=user_data.get('birthday', 'N/A'), vector_count=user_data.get('vector_count', 0), f_data=f_data)
                                self._send_user_webhook(user_id, user_name, status, is_unknown=False)

                            if user_data.get('vector_count', 0) < 10:
                                self._auto_learn_face(frame, face, user_id, user_name, user_data.get('birthday', 'N/A'), target_cid, attendance_mgr)
                            f_data['unknown_attempts'] = 0
                    else:
                        f_data['unknown_attempts'] += 1
                        f_data['last_attempt_time'] = current_time
                        f_data['user_data'] = vote
                        # --- CƠ CHẾ CHỐNG SPAM NHƯNG VẪN REALTIME ---
                        # Chỉ phát báo động BẰNG GIỌNG NÓI khi thực sự đã phân tích 10 lần liên tiếp thất bại.
                        # Vẫn giữ Rate-Limiter 5 giây 1 lần để tránh loa kêu dồn dập.
                        if f_data['unknown_attempts'] >= 10:
                            last_unknown = f_data.get('last_unknown_alert', 0)
                            if current_time - last_unknown > 5.0:
                                f_data['last_unknown_alert'] = current_time
                                logger.warning(f"[Recognition] ❌ Unknown (10+ attempts) - Triggering Alert")
                                
                                # Cấp phát tên video cho người lạ
                                if f_data.get('video_path') is None:
                                    from src.config import CAPTURES_DIR
                                    v_dir = CAPTURES_DIR / "Unknown_Videos"
                                    v_dir.mkdir(parents=True, exist_ok=True)
                                    v_path = v_dir / f"{int(time.time())}_Nguoi_La.mp4"
                                    f_data['video_path'] = str(v_path)
                                
                                self._send_telegram_alert("PHAT HIEN NGUOI LA", f"Phát hiện người lạ trước camera (Đã quét {f_data['unknown_attempts']} lần)", image=frame)
                                self._send_user_webhook("Unknown", "Nguoi la", "unknown", is_unknown=True, custom_voice_text="Xin vui lòng thử lại")

                        f_data['status'] = 'RETRY_WAIT'
                else:
                    # Still gathering
                    f_data['status'] = 'GATHERING'
                    face.name = f"Dang phan tich... ({f_data['gather_count']}/{GATHER_FRAMES})"
                    face.score = 0.0
            
            updated_faces_map[matched_id] = f_data

            if matched_id in updated_faces_map:
                data = updated_faces_map[matched_id]
                # Flag cho draw_faces() biết màu nào cần tô
                face.recognized = data['status'] in ['RECOGNIZED', 'COOLDOWN']
                face.is_spoof   = data['status'] == 'SPOOF_DETECTED'

                if data['user_data']:
                    u_d = data['user_data']
                    face.user_id = u_d.get('user_id') or 'Unknown'
                    face.score = u_d.get('score') or 0.0
                    face.birthday = u_d.get('birthday') or 'N/A'
                    face.detect_time = (u_d.get('detect_time') or '').split(' ')[-1] or 'N/A'
                    face.vector_count = u_d.get('vector_count') if u_d.get('vector_count') is not None else 0
                    
                    if data['status'] in ['COOLDOWN', 'RECOGNIZED']:
                        name = remove_accents(u_d.get('name'))
                        uid = u_d.get('user_id')
                        if uid in self.user_cooldowns:
                            elapsed = current_time - self.user_cooldowns[uid]
                            remain_sec = int(max(0, RecognitionConfig.COOLDOWN_SECONDS - elapsed))
                            if remain_sec > 0:
                                if remain_sec >= 60:
                                    face.name = f"{name} (Cho: {remain_sec // 60}p {remain_sec % 60}s)"
                                else:
                                    face.name = f"{name} (Cho: {remain_sec}s)"
                            else:
                                face.name = f"{name} (Da san sang)"
                        else:
                            face.name = f"{name} (Dang Cho)"
                    elif data['status'] == 'RETRY_WAIT':
                        face.name = f"Chua ro (Dang quet lai...)"
                    elif data['status'] == 'UNAUTHORIZED':
                        face.name = "!!! TRUY CAP LA !!!"
                    elif data['status'] == 'SPOOF_DETECTED':
                        face.name = "!!! GIA MAO !!!"
                    else:
                        face.name = remove_accents(u_d.get('name', 'Chua ro'))
                else:
                    if data['status'] == 'GATHERING':
                        count = data.get('gather_count', 0)
                        total = getattr(RecognitionConfig, 'GATHER_FRAMES', 5)
                        face.name = f"Dang phan tich... ({count}/{total})"
                    elif data['status'] == 'RETRY_WAIT':
                        face.name = f"Chua nhan ra (Dang quet lai...)"
                    elif data['status'] == 'UNAUTHORIZED':
                        face.name = "Nguoi la"
                    else:
                        face.name = "Dang phan tich..."
                    face.score = 0.0

        # ─── SYNC & CLEANUP ─────────────────────────────────────────────────
        # 1. Cập nhật self.active_faces với các thay đổi trong frame này
        for fid, fdata in updated_faces_map.items():
            self.active_faces[fid] = fdata

        # 2. Xóa các face ID không xuất hiện trong vòng 1 giây & XUẤT VIDEO
        new_active_faces = {}
        for fid, fdata in self.active_faces.items():
            if current_time - fdata['last_seen'] < 1.0:
                new_active_faces[fid] = fdata
            else:
                # Ngươi này đã rời đi: tiến hành ghi Video đối soát nếu có
                v_path = fdata.get('video_path')
                v_frames = fdata.get('video_frames', [])
                if v_path and len(v_frames) > 5:
                    # Tính toán FPS thực tế dựa trên thời gian xuất hiện
                    duration = current_time - fdata.get('start_time', current_time)
                    actual_fps = len(v_frames) / duration if duration > 0 else 15.0
                    actual_fps = max(5.0, min(30.0, actual_fps)) # Giới hạn FPS hợp lý
                    
                    def save_video(path, frames, fps):
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            h, w = frames[0].shape[:2]
                            out = cv2.VideoWriter(path, fourcc, fps, (w, h))
                            for vf in frames:
                                out.write(vf)
                            out.release()
                        except Exception as e:
                            logger.error(f"Cannot save tracker video: {e}")
                    threading.Thread(target=save_video, args=(v_path, v_frames, actual_fps), daemon=True).start()

        self.active_faces = new_active_faces
        
        # 3. Cleanup spoof sync records cho các face đã biến mất hoàn toàn
        if self.spoof_checker:
            self.spoof_checker.cleanup_stale(set(self.active_faces.keys()))

