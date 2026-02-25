import time
import numpy as np
from loguru import logger
from typing import List, Dict, Any
import cv2
import requests
import threading
from src.config import RecognitionConfig, CAPTURES_DIR, ApiConfig, CameraConfig, WebhookConfig
from src.attendance.mongodb_mgr import mongo_db
from src.utils.string_utils import remove_accents
from src.utils.time_manager import time_mgr

class FaceTracker:
    """
    Tracks faces across frames to ensure stability before recognition.
    Includes a cooldown mechanism and unknown-face attempt logic with audio.
    """
    def __init__(self, threshold_seconds=1.5):
        self.active_faces: Dict[int, Dict[str, Any]] = {} # {id: data}
        self.face_id_counter = 0
        self.threshold_seconds = threshold_seconds
        
        # Global cooldowns: {user_id: last_detection_time}
        self.user_cooldowns: Dict[str, float] = {}
        
        # Webhook cooldown for known users: {user_id: last_webhook_time}
        # Prevents sending duplicate webhooks within 15 minutes
        self.webhook_sent_time: Dict[str, float] = {}
        
        # Unknown face webhook counter (max 10, reset on successful checkin)
        self.unknown_webhook_count = 0
        self.unknown_webhook_limit = 10

    def _send_user_webhook(self, user_id, user_name, status, is_unknown=False):
        """
        Sends user detection info to the configured webhook URL in a background thread.
        
        Deduplication rules:
        - Known users: Max 1 webhook per 15 minutes per user_id
        - Unknown faces: Max 10 webhooks total, reset when any known user checks in
        """
        def thread_task():
            try:
                from src.utils.string_utils import remove_accents
                from src.utils.time_manager import time_mgr
                
                current_time = time.time()
                
                # === DEDUPLICATION LOGIC ===
                if is_unknown:
                    # Check if unknown webhook limit reached
                    if self.unknown_webhook_count >= self.unknown_webhook_limit:
                        logger.debug(f"Unknown webhook limit reached ({self.unknown_webhook_count}/{self.unknown_webhook_limit}). Skipping.")
                        return
                    
                    # Increment counter
                    self.unknown_webhook_count += 1
                    logger.warning(f"Sending unknown webhook #{self.unknown_webhook_count}/{self.unknown_webhook_limit}")
                else:
                    # Known user - check 15 minute cooldown (Database Persistent + Local Memory)
                    if status == 'COOLDOWN':
                        # Allow webhook to proceed but log it. 
                        # Deduplication (15m) will still apply.
                        logger.debug(f"Handling COOLDOWN webhook for {user_name}")

                    if user_id in self.webhook_sent_time:
                        last_sent = self.webhook_sent_time[user_id]
                        elapsed = current_time - last_sent
                        
                        # Only block if it's NOT a COOLDOWN status. 
                        # We want the "Ban da truy cap gan day" voice feedback to work immediately.
                        if status != 'COOLDOWN' and elapsed < 900:  # 15 minutes = 900 seconds
                            remaining = int(900 - elapsed)
                            logger.debug(f"Webhook blocked for {user_name} (sent {int(elapsed)}s ago, {remaining}s remaining)")
                            return
                    
                    # Record this webhook send
                    self.webhook_sent_time[user_id] = current_time
                    
                    # Reset unknown counter when a known user checks in successfully
                    if self.unknown_webhook_count > 0:
                        logger.info(f"Resetting unknown webhook counter (was {self.unknown_webhook_count})")
                        self.unknown_webhook_count = 0
                    
                    logger.info(f"Webhook allowed for {user_name}")
                
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
                elif status == "COOLDOWN":
                    voice_text = f"Bạn {user_name} đã truy cập gần đây"
                elif status == "SPOOF":
                    voice_text = "Cảnh báo: Phát hiện hành vi giả mạo khuôn mặt"
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
                    logger.debug(f"Webhook sent successfully for {user_name}")
                else:
                    logger.warning(f"Webhook failed for {user_name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error sending webhook: {e}")

        # Run in background to not block the tracking loop
        threading.Thread(target=thread_task, daemon=True).start()

    def _get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def _save_log_with_bbox(self, frame, face, user_id, user_name, score, is_known=True, status=None, company_id=None, unknown_attempt=0, birthday="N/A", vector_count=0):
        """Sync state check + Async heavy saving."""
        if frame is None: return None, None
        
        from src.attendance.mongodb_mgr import mongo_db
        # 1. IMMEDIATE SYNC CHECK (Fast)
        final_status = status
        if is_known and status is None:
            # Important: pass company_id to check the correct log collection
            final_status = mongo_db.get_attendance_status(user_id, company_id=company_id) 
            
        # --- BLOCK NEW SAVES IF IN COOLDOWN ---
        if final_status == 'COOLDOWN':
            logger.debug(f"Bỏ qua lưu ảnh/log cho {user_id} ({user_name}) vì đang trong thời gian chặn (Cooldown).")
            return None, final_status
        
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
                        f"N-sinh: {birthday}",
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

                # Tên ảnh chỉ cần timestamp vì đã nằm trong thư mục tên
                img_name = f"{int(time.time())}.webp"
                img_path = str(target_dir / img_name)
                
                # Heavy Disk Write with quality 50 to target ~10KB
                cv2.imwrite(img_path, final_img, [int(cv2.IMWRITE_WEBP_QUALITY), 50])
                # Actual DB Commit
                mongo_db.log_attendance(user_id, user_name, status=status, frame=final_img, company_id=company_id, unknown_attempt=unknown_attempt)
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
            for f_id, f_data in self.active_faces.items():
                if f_id in used_ids_in_frame: continue
                prev_center = f_data['center']
                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                if dist < 80: 
                    potential_matches.append((dist, f_id))
            
            if potential_matches:
                potential_matches.sort()
                matched_id = potential_matches[0][1]
                used_ids_in_frame.add(matched_id)
            
            if matched_id is None:
                matched_id = self.face_id_counter
                self.face_id_counter += 1
                
                # Initial status: FORCED TO STABILIZING FOR BYPASS
                initial_status = 'STABILIZING'
                
                updated_faces_map[matched_id] = {
                    'start_time': current_time,
                    'last_seen': current_time,
                    'center': center,
                    'status': initial_status,
                    'user_data': None,
                    'cooldown_remaining': 0,
                    'unknown_attempts': 0,
                    'last_attempt_time': 0,
                    'liveness_verified': False
                }
            else:
                f_data = self.active_faces[matched_id]
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
                
                # Update status if not in a final state
                if f_data['status'] not in ['RECOGNIZED', 'COOLDOWN', 'RETRY_WAIT', 'UNAUTHORIZED', 'SPOOF_DETECTED']:
                    f_data['status'] = 'STABILIZING'

                time_stayed = current_time - f_data['start_time']
                wait_time = current_time - f_data['last_attempt_time']
                can_attempt = False
                
                # Check if we can attempt recognition, but ONLY if none done this frame yet
                # Note: Already recognized or cooldown faces do NOT block others
                if not is_real or f_data['status'] in ['RECOGNIZED', 'COOLDOWN', 'SPOOF_DETECTED']:
                    can_attempt = False
                elif not recognition_done_this_frame:
                    if f_data['status'] == 'STABILIZING':
                        if time_stayed >= 1.5:  # Increased from 0.5s to 1.5s for better stability
                            can_attempt = True
                    elif f_data['status'] in ['RETRY_WAIT', 'UNAUTHORIZED']:
                        if f_data['unknown_attempts'] < 5 or wait_time >= 5.0:
                            can_attempt = True
                
                # ... (rest of recognition logic)

                if can_attempt:
                    if face_rec is None or attendance_mgr is None:
                        f_data['status'] = 'MONITORING'
                        continue
                    
                    recognition_done_this_frame = True # Rate limit
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    
                    # 1. RUN ANTI-SPOOFING (Sequential check after stability)
                    logger.debug(f"Triggering Anti-Spoofing for ID: {matched_id}")
                    as_label, as_score = face_rec.anti_spoof.predict(frame, face)
                    
                    face.is_real = (as_label == 1)
                    face.as_label = int(as_label)
                    face.as_score = float(as_score)
                    f_data['liveness_verified'] = True
                    
                    if not face.is_real:
                        # ... (spoof logic)
                        f_data['status'] = 'SPOOF_DETECTED'
                        if f_data.get('last_spoof_log', 0) < current_time - 30:
                            self._save_log_with_bbox(frame, face, "Spoof", "Kẻ giả mạo", 
                                                    score=face.as_score, is_known=False, 
                                                    status="SPOOF", company_id=active_company)
                            self._send_user_webhook("Spoof", "Kẻ giả mạo", "SPOOF", is_unknown=True)
                            f_data['last_spoof_log'] = current_time
                        logger.warning(f"SPOOF DETECTED for ID {matched_id}")
                    else:
                        # CASE: REAL FACE -> PROCEED TO RECOGNITION
                        logger.debug(f"Face is REAL. Extracting embedding for ID: {matched_id}")
                        face_rec.rec_model.get(frame, face) # Extract 512-dim embedding
                        
                        user_data = attendance_mgr.recognize(face.normed_embedding)
                        user_data['as_score'] = face.as_score # Keep the score for sync back
                        user_id = user_data.get('user_id', 'Unknown')
                        user_name = user_data.get('name', 'Unknown')

                        if only_recognize:
                            f_data['status'] = 'RECOGNIZED'
                            f_data['user_name'] = user_name
                            f_data['user_id'] = user_id
                            f_data['user_data'] = user_data
                        elif user_name != "Unknown":
                            target_cid = user_data.get('company_id') or active_company
                            if not RecognitionConfig.TEST_MODE and user_id in self.user_cooldowns:
                                elapsed = current_time - self.user_cooldowns[user_id]
                                if elapsed < RecognitionConfig.COOLDOWN_SECONDS:
                                    f_data['status'] = 'COOLDOWN'
                                    f_data['user_data'] = user_data
                                    f_data['cooldown_remaining'] = int(RecognitionConfig.COOLDOWN_SECONDS - elapsed)
                                    self._send_user_webhook(user_id, user_name, 'COOLDOWN', is_unknown=False)
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
                            
                            # Auto-enrichment
                            vector_count = user_data.get('vector_count', 0)
                            if vector_count < 10:
                                attendance_mgr.upsert_user(user_name=user_name, user_id=user_id, birthday=user_data.get('birthday', 'N/A'), embeddings=[face.normed_embedding], company_id=target_cid)
                            
                            f_data['unknown_attempts'] = 0
                        else:
                            f_data['unknown_attempts'] += 1
                            f_data['last_attempt_time'] = current_time
                            f_data['user_data'] = user_data
                            self._send_user_webhook("Unknown", f"Nguoi la {f_data['unknown_attempts']}", "unknown", is_unknown=True)
                            self._save_log_with_bbox(frame, face, "Unknown", "Người lạ", score=0.0, is_known=False, status="FAILED", company_id=active_company, unknown_attempt=f_data['unknown_attempts'])
                            
                            if f_data['unknown_attempts'] >= 10:
                                f_data['status'] = 'UNAUTHORIZED'
                            else:
                                f_data['status'] = 'RETRY_WAIT'
                        
                        # Các lần thử khác chỉ log console để theo dõi
                        logger.warning(f"Unknown face. Attempt {f_data['unknown_attempts']} in progress.")
                
                updated_faces_map[matched_id] = f_data

            # Update visualization attributes
            if matched_id in updated_faces_map:
                data = updated_faces_map[matched_id]
                if data['user_data']:
                    u_d = data['user_data']
                    face.user_id = u_d.get('user_id') or 'Unknown'
                    face.score = u_d.get('score') or 0.0
                    face.birthday = u_d.get('birthday') or 'N/A'
                    face.detect_time = (u_d.get('detect_time') or '').split(' ')[-1] or 'N/A'
                    face.vector_count = u_d.get('vector_count') if u_d.get('vector_count') is not None else 0
                    
                    if data['status'] == 'COOLDOWN':
                        name = remove_accents(u_d.get('name'))
                        face.name = f"{name} (Khoa {data['cooldown_remaining'] // 60}p)"
                    elif data['status'] == 'RETRY_WAIT':
                        wait_left = int(5 - (current_time - data['last_attempt_time']))
                        face.name = f"Chua ro (Thu lai {max(0, wait_left)}s)"
                    elif data['status'] == 'UNAUTHORIZED':
                        face.name = "!!! TRUY CAP LAI !!!"
                    elif data['status'] == 'SPOOF_DETECTED':
                        face.name = "!!! CANH BAO GIA MAO!!!"
                    else:
                        face.name = remove_accents(u_d.get('name', 'Chua ro'))
                else:
                    if data['status'] == 'SPOOF_DETECTED':
                        face.name = "MAT GIA / SPOOF"
                    elif data['status'] == 'LIVENESS_WAITING':
                        face.name = "VUI LONG NHAY MAT..."
                    else:
                        face.name = "Dang phan tich..."
                    face.score = 0.0

        self.active_faces = {k: v for k, v in updated_faces_map.items() if current_time - v['last_seen'] < 1.0}

