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

    def _send_user_webhook(self, user_id, user_name, status, is_unknown=False):
        """
        Sends user detection info to the configured webhook URL in a background thread.
        
        Deduplication rules:
        - Known users: Max 1 webhook per 15 minutes per user_id
        - Unknown faces: Max 10 webhooks total, reset when any known user checks in
        """
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
                if status in ["IN", "OUT"]:
                    action_vn = "vào" if status == "IN" else "ra"
                    voice_text = f"Xin chào {user_name}, bạn đã chấm công {action_vn} thành công"
                elif status == "SPOOF":
                    voice_text = "Cảnh báo: Phát hiện hành vi giả mạo khuôn mặt"
                elif status == "COOLDOWN":
                    voice_text = f"{user_name} đã truy cập gần đây"
                else:
                    # Default for unknown/unauthorized
                    voice_text = "Xin vui lòng thử lại"

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

    def _save_log_with_bbox(self, frame, face, user_id, user_name, score, is_known=True, status=None, company_id=None, unknown_attempt=0, birthday="N/A", vector_count=0):
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
                # This meets user request for ~10KB file size
                final_img = cv2.resize(save_frame, (640, 360), interpolation=cv2.INTER_AREA)

                raw_company_name = mongo_db.get_company_name(company_id) if company_id else "unknown_company"
                company_folder = remove_accents(raw_company_name).replace(" ", "_")
                
                # Tạo thư mục con theo tên người (thư mục tên)
                if is_known:
                    user_folder = f"{remove_accents(user_name).replace(' ','_')}_{user_id}"
                    target_dir = CAPTURES_DIR / company_folder / user_folder
                else:
                    target_dir = CAPTURES_DIR / company_folder / "Nguoi_La"
                
                target_dir.mkdir(parents=True, exist_ok=True)

                import time
                # Tên ảnh chỉ cần timestamp vì đã nằm trong thư mục tên
                img_name = f"{int(time.time())}.webp"
                img_path = str(target_dir / img_name)
                
                # Heavy Disk Write with quality 50 to target ~10KB
                cv2.imwrite(img_path, final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 50])
                # Actual DB Commit
                mongo_db.log_attendance(user_id, user_name, status=final_status, frame=final_img, company_id=company_id, unknown_attempt=unknown_attempt)
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

        for face in detected_faces:
            center = self._get_center(face.bbox)
            matched_id = None
            is_real = getattr(face, 'is_real', True)
            as_label = getattr(face, 'as_label', 1) # 0=SPOOF, 1=REAL, 2=WAITING
            
            # ... (matching logic remains the same)
            # Sort active faces by distance to current center to find the best match first
            potential_matches = []
            
            # Tính giới hạn khoảng cách (dynamic threshold) dựa trên kích thước khuôn mặt thực tế
            face_w = face.bbox[2] - face.bbox[0]
            face_h = face.bbox[3] - face.bbox[1]
            # Mở rộng giới hạn lên tối thiểu 250 pixels, hoặc gấp 1.5 lần size mặt (nếu mặt quá to do đứng gần)
            max_dist = max(250, max(face_w, face_h) * 1.5)

            for f_id, f_data in self.active_faces.items():
                if f_id in used_ids_in_frame: continue
                prev_center = f_data['center']
                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                
                if dist < max_dist: 
                    potential_matches.append((dist, f_id))
            
            if potential_matches:
                potential_matches.sort()
                matched_id = potential_matches[0][1]
                used_ids_in_frame.add(matched_id)
            
            if matched_id is None:
                matched_id = self.face_id_counter
                self.face_id_counter += 1
                
                # Trạng thái khởi đầu: GATHERING — thu thập embedding từ nhiều frame
                # trước khi kết luận để tránh sai do mặt nhìn từ xa/mờ
                updated_faces_map[matched_id] = {
                    'start_time':         current_time,
                    'last_seen':          current_time,
                    'center':             center,
                    'status':             'GATHERING',
                    'user_data':          None,
                    'cooldown_remaining': 0,
                    'unknown_attempts':   0,
                    'last_attempt_time':  0,
                    # === EMBEDDING ACCUMULATOR ===
                    # Lưu raw embedding mỗi frame, cuối gọi Qdrant 1 lần duy nhất
                    'gathering_embeddings': [],  # List[np.ndarray] 512D
                    'gather_count':         0,
                }
            else:
                f_data = self.active_faces[matched_id]
                # Preserve the force_immediate_attempt flag if it exists, otherwise False
                if 'force_immediate_attempt' not in f_data:
                    f_data['force_immediate_attempt'] = False
                    
                f_data['last_seen'] = current_time
                f_data['center'] = center

                # SYNC BACK: If we already verified this face in previous frames, 
                # update the current frame's face object to stop it from showing "Analyzing"
                if f_data.get('liveness_verified'):
                    face.as_label = 1 if f_data['status'] != 'SPOOF_DETECTED' else 0
                    face.is_real = (face.as_label == 1)
                    if f_data['user_data']:
                        face.name = remove_accents(f_data['user_data'].get('name', 'Unknown'))
                        face.as_score = f_data['user_data'].get('as_score', 1.0)
                        pass
                
                # Update status if not in a final state
                if f_data['status'] not in ['RECOGNIZED', 'COOLDOWN', 'RETRY_WAIT', 'UNAUTHORIZED', 'SPOOF_DETECTED']:
                    pass # Don't rewrite status. We handle recognition via the force_immediate flag now.

                # =======================================================
                # CAN_ATTEMPT LOGIC
                # =======================================================
                # Thời gian kể từ lần thử cuối (dùng cho RETRY_WAIT cooldown)
                wait_time = current_time - f_data.get('last_attempt_time', 0)

                # Chỉ bỏ qua khi đã có kết luận cuối (RECOGNIZED/COOLDOWN)
                can_attempt = False
                if f_data['status'] in ['RECOGNIZED', 'COOLDOWN']:
                    can_attempt = False
                elif f_data['status'] == 'GATHERING':
                    can_attempt = True
                elif f_data['status'] == 'STABILIZING':
                    can_attempt = True
                    f_data['status'] = 'GATHERING'
                    f_data.setdefault('gathering_votes', [])
                    f_data.setdefault('gather_count', 0)
                elif f_data['status'] in ['RETRY_WAIT', 'UNAUTHORIZED', 'SPOOF_DETECTED']:
                    # Reset sau 3s nếu là unknown/unauth, hoặc 5s nếu là spoof để khắt khe hơn
                    retry_delay = 5.0 if f_data['status'] == 'SPOOF_DETECTED' else 3.0
                    if wait_time >= retry_delay:
                        f_data['status'] = 'GATHERING'
                        f_data['gathering_embeddings'] = []  # reset accumulator
                        f_data['gather_count'] = 0
                        can_attempt = True

                if can_attempt:
                    if face_rec is None or attendance_mgr is None:
                        f_data['status'] = 'MONITORING'
                        continue

                    # ———————————————————————————
                    # BEST-OF-N GATHERING: Thu thập GATHER_FRAMES embedding
                    # rồi mới kết luận — tránh sai khi mặt nhìn từ xa/mờ
                    # ———————————————————————————
                    GATHER_FRAMES = getattr(RecognitionConfig, 'GATHER_FRAMES', 5)

                    # Bước 1: Extract ArcFace embedding (không gọi Qdrant ở đây)
                    logger.debug(f"[Gathering] ID={matched_id} frame {f_data.get('gather_count',0)+1}/{GATHER_FRAMES}")
                    face_rec.rec_model.get(frame, face)
                    emb = face.normed_embedding.copy()  # unit-norm 512D vector
                    # (Spoof check sẽ chạy ĐỒNG BỘ tại bước kết luận — không async submit nữa)

                    # Bước 2: Lưu embedding vào accumulator
                    f_data.setdefault('gathering_embeddings', [])
                    f_data['gathering_embeddings'].append(emb)
                    f_data['gather_count'] = len(f_data['gathering_embeddings'])

                    # Bước 3: Chưa đủ frame → tiếp tục GATHERING (không tốn Qdrant)
                    if f_data['gather_count'] < GATHER_FRAMES:
                        f_data['status'] = 'GATHERING'
                        face.name = f"Dang phan tich... ({f_data['gather_count']}/{GATHER_FRAMES})"
                        face.score = 0.0
                    else:
                        # Bước 4: Đủ GATHER_FRAMES → Average embeddings → 1 Qdrant query
                        embs = np.array(f_data['gathering_embeddings'])   # shape (N, 512)
                        avg_emb = embs.mean(axis=0)                       # average
                        norm = np.linalg.norm(avg_emb)
                        if norm > 0:
                            avg_emb = avg_emb / norm                      # L2 normalize

                        logger.debug(
                            f"[Gathering] ID={matched_id}: {GATHER_FRAMES} embeddings averaged "
                            f"→ 1 Qdrant query (noise reduced)"
                        )
                        vote = attendance_mgr.recognize(avg_emb)  # 1 network call thay vì 3

                        # Reset accumulator cho lần tiếp theo
                        f_data['gathering_embeddings'] = []
                        f_data['gather_count'] = 0

                        # Bước 5: Kết luận từ 1 vote của averaged embedding
                        is_known = vote.get('name', 'Unknown') != 'Unknown'

                        if is_known:
                            # ── Anti-Spoof check ĐỒNG BỘ — chạy ngay, có kết quả chắc chắn ──
                            # check_sync() dùng frame + face hiện tại, timeout 150ms
                            if self.spoof_checker and frame is not None:
                                spoof_real, spoof_score = self.spoof_checker.check_sync(
                                    frame, face, timeout=0.15
                                )
                                if not spoof_real:
                                    logger.warning(f"[Spoof] ❌ GIẢ MẠO phát hiện — ID={matched_id} score={spoof_score:.2f}")
                                    f_data['status'] = 'SPOOF_DETECTED'
                                    f_data['last_attempt_time'] = current_time # Gán để cooldown retry
                                    self._send_user_webhook("Spoof", "Ke gia mao", "SPOOF", is_unknown=True)
                                    
                                    # Send Telegram report for Spoofing
                                    from src.utils.telegram_bot import send_telegram_report
                                    send_telegram_report(
                                        "Spoof Detected", 
                                        f"Phát hiện hành vi giả mạo!\nID: {matched_id}\nSpoof Score: {spoof_score:.2f}",
                                        image=frame
                                    )
                                    
                                    updated_faces_map[matched_id] = f_data
                                    continue  # Không log chấm công
                            # ── RECOGNIZED ──────────────────────────────────────
                            user_data = vote
                            user_id   = user_data.get('user_id', 'Unknown')
                            user_name = user_data.get('name', 'Unknown')
                            logger.info(f"[Gathering] ✅ {user_name} (avg score={vote.get('score',0):.3f}, {GATHER_FRAMES} frames)")

                            if only_recognize:
                                f_data['status']    = 'RECOGNIZED'
                                f_data['user_name'] = user_name
                                f_data['user_id']   = user_id
                                f_data['user_data'] = user_data
                            else:
                                target_cid = user_data.get('company_id') or active_company
                                if not RecognitionConfig.TEST_MODE and user_id in self.user_cooldowns:
                                    elapsed = current_time - self.user_cooldowns[user_id]
                                    if elapsed < RecognitionConfig.COOLDOWN_SECONDS and RecognitionConfig.COOLDOWN_SECONDS > 0:
                                        f_data['status'] = 'COOLDOWN'
                                        f_data['user_data'] = user_data
                                        f_data['cooldown_remaining'] = int(RecognitionConfig.COOLDOWN_SECONDS - elapsed)
                                        self._send_user_webhook(user_id, user_name, "COOLDOWN", is_unknown=False)
                                    else:
                                        f_data['status'] = 'RECOGNIZED'
                                        f_data['user_data'] = user_data
                                        self.user_cooldowns[user_id] = current_time
                                        url, status = self._save_log_with_bbox(frame, face, user_id, user_name, user_data.get('score', 0.0),
                                                                               company_id=target_cid, birthday=user_data.get('birthday', 'N/A'), vector_count=user_data.get('vector_count', 0))
                                        self._send_user_webhook(user_id, user_name, status, is_unknown=False)
                                else:
                                    f_data['status'] = 'RECOGNIZED'
                                    f_data['user_data'] = user_data
                                    self.user_cooldowns[user_id] = current_time
                                    url, status = self._save_log_with_bbox(frame, face, user_id, user_name, user_data.get('score', 0.0),
                                                                           company_id=target_cid, birthday=user_data.get('birthday', 'N/A'), vector_count=user_data.get('vector_count', 0))
                                    self._send_user_webhook(user_id, user_name, status, is_unknown=False)

                                # Auto-enrich nếu chưa đủ mẫu
                                if user_data.get('vector_count', 0) < 10:
                                    self._auto_learn_face(frame, face, user_id, user_name, user_data.get('birthday', 'N/A'), target_cid, attendance_mgr)
                                    
                                f_data['unknown_attempts'] = 0

                        else:
                            # Unknown → "Xin vui lòng thử lại"
                            f_data['unknown_attempts'] += 1
                            f_data['last_attempt_time'] = current_time
                            f_data['user_data'] = vote
                            logger.warning(f"[Gathering] ❌ Unknown avg embedding (lần {f_data['unknown_attempts']}/5).")
                            self._send_user_webhook("Unknown", "Nguoi la", "unknown", is_unknown=True)
                            if f_data['unknown_attempts'] >= 5:
                                f_data['status'] = 'UNAUTHORIZED'
                            else:
                                f_data['status'] = 'RETRY_WAIT'

                
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
                        wait_left = int(2 - (current_time - data['last_attempt_time']))
                        face.name = f"Chua ro (Thu lai {max(0, wait_left)}s)"
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
                        wait_left = int(3 - (current_time - data['last_attempt_time']))
                        attempts = data.get('unknown_attempts', 0)
                        face.name = f"Chua nhan ra (lan {attempts}/5, thu lai sau {max(0, wait_left)}s)"
                    elif data['status'] == 'UNAUTHORIZED':
                        face.name = "Nguoi la"
                    else:
                        face.name = "Dang phan tich..."
                    face.score = 0.0

        # ─── SYNC & CLEANUP ─────────────────────────────────────────────────
        # 1. Cập nhật self.active_faces với các thay đổi trong frame này
        for fid, fdata in updated_faces_map.items():
            self.active_faces[fid] = fdata

        # 2. Xóa các face ID không xuất hiện trong vòng 1 giây (để giữ ổn định khi mất frame)
        self.active_faces = {
            k: v for k, v in self.active_faces.items() 
            if current_time - v['last_seen'] < 1.0
        }
        
        # 3. Cleanup spoof sync records cho các face đã biến mất hoàn toàn
        if self.spoof_checker:
            self.spoof_checker.cleanup_stale(set(self.active_faces.keys()))

