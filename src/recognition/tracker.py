import time
import numpy as np
from loguru import logger
from typing import List, Dict, Any
import cv2
from src.config import RecognitionConfig, CAPTURES_DIR, ApiConfig, CameraConfig
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

    def _get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def _save_log_with_bbox(self, frame, face, user_id, user_name, score, is_known=True, status=None, company_id=None):
        """
        Draws a bounding box and saves/logs the frame.
        """
        if frame is None:
            return None, None

        # Draw box on a copy
        annotated_frame = frame.copy()
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if is_known else (0, 255, 255) # Green for known, Yellow for unknown
        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Add label
        user_name_no_accents = remove_accents(user_name)
        label = f"{user_name_no_accents} ({score:.2f})"
        cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add timestamp and camera name overlay
        ts, _ = time_mgr.get_formatted_time()
        overlay_text = f"{CameraConfig.CAMERA_NAME} | {ts}"
        # Draw at bottom right
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = annotated_frame.shape[1] - tw - 10
        ty = annotated_frame.shape[0] - 10
        
        # Draw background for better visibility
        cv2.rectangle(annotated_frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
        cv2.putText(annotated_frame, overlay_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Save to local captures
        current_time = time.time()
        img_prefix = user_id if is_known else "unknown"
        img_name = f"{img_prefix}_{int(current_time)}.jpg"
        img_path = str(CAPTURES_DIR / img_name)
        cv2.imwrite(img_path, annotated_frame)
        
        # Log to MongoDB with the annotated frame
        res_status = mongo_db.log_attendance(user_id, user_name, status=status, frame=annotated_frame, company_id=company_id)
        
        url = f"{ApiConfig.BASE_URL}/captures/{img_name}"
        return url, res_status

    def update(self, detected_faces, attendance_mgr, frame=None):
        """
        Assigns IDs and decides when to trigger recognition or alerts.
        """
        current_time = time.time()
        updated_faces_map = {}
        
        for face in detected_faces:
            center = self._get_center(face.bbox)
            matched_id = None
            
            for f_id, f_data in self.active_faces.items():
                prev_center = f_data['center']
                dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                if dist < 50:
                    matched_id = f_id
                    break
            
            if matched_id is None:
                matched_id = self.face_id_counter
                self.face_id_counter += 1
                updated_faces_map[matched_id] = {
                    'start_time': current_time,
                    'last_seen': current_time,
                    'center': center,
                    'status': 'STABILIZING',
                    'user_data': None,
                    'cooldown_remaining': 0,
                    'unknown_attempts': 0,
                    'last_attempt_time': 0
                }
            else:
                f_data = self.active_faces[matched_id]
                f_data['last_seen'] = current_time
                f_data['center'] = center
                
                time_stayed = current_time - f_data['start_time']
                
                # Logic xác định khi nào được phép nhận diện
                wait_time = current_time - f_data['last_attempt_time']
                can_attempt = False
                
                if f_data['status'] == 'STABILIZING':
                    # Người mới: Cần đủ thời gian ổn định (2.0s)
                    if time_stayed >= self.threshold_seconds:
                        can_attempt = True
                elif f_data['status'] == 'RETRY_WAIT':
                    if f_data['unknown_attempts'] < 5:
                        # Chu kỳ 1: Thử lại liên tục 5 lần đầu (không delay)
                        can_attempt = True
                    elif f_data['unknown_attempts'] < 10:
                        # Chu kỳ 2: Sau 5 lần xịt, nghỉ 5s rồi cho thử tiếp 5 lần nữa
                        if wait_time >= 5.0:
                            can_attempt = True

                if can_attempt:
                    user_data = attendance_mgr.recognize(face.normed_embedding)
                    user_id = user_data.get('user_id', 'Unknown')
                    user_name = user_data.get('name', 'Unknown')
                    
                    if user_name != "Unknown":
                        # CASE: THÀNH CÔNG
                        if not RecognitionConfig.TEST_MODE and user_id in self.user_cooldowns:
                            elapsed = current_time - self.user_cooldowns[user_id]
                            if elapsed < RecognitionConfig.COOLDOWN_SECONDS:
                                f_data['status'] = 'COOLDOWN'
                                f_data['user_data'] = user_data
                                f_data['cooldown_remaining'] = int(RecognitionConfig.COOLDOWN_SECONDS - elapsed)
                            else:
                                f_data['status'] = 'RECOGNIZED'
                                f_data['user_data'] = user_data
                                self.user_cooldowns[user_id] = current_time
                                _, status = self._save_log_with_bbox(frame, face, user_id, user_name, user_data.get('score', 0.0), company_id=user_data.get('company_id'))
                                
                                if status == 'IN': logger.info(f"Diem danh VAO: {user_name}")
                                elif status == 'OUT': logger.info(f"Diem danh RA: {user_name}")
                                else: logger.info(f"Diem danh THANH CONG: {user_name}")
                        else:
                            f_data['status'] = 'RECOGNIZED'
                            f_data['user_data'] = user_data
                            self.user_cooldowns[user_id] = current_time
                            _, status = self._save_log_with_bbox(frame, face, user_id, user_name, user_data.get('score', 0.0), company_id=user_data.get('company_id'))
                            
                            if status == 'IN': logger.info(f"Diem danh VAO: {user_name}")
                            elif status == 'OUT': logger.info(f"Diem danh RA: {user_name}")
                            else: logger.info(f"Diem danh THANH CONG: {user_name}")
                        
                        f_data['unknown_attempts'] = 0 # Reset khi thành công
                    
                    else:
                        # CASE: KHÔNG NHẬN DIÊN ĐƯỢC (UNKNOWN)
                        f_data['unknown_attempts'] += 1
                        f_data['last_attempt_time'] = current_time
                        f_data['user_data'] = user_data
                        f_data['status'] = 'RETRY_WAIT'
                        
                        # Ghi log lỗi vào DB tại các mốc quan trọng
                        if f_data['unknown_attempts'] == 5:
                            # Sau 5 lần đầu thất bại: Ghi log 1 lần
                            self._save_log_with_bbox(frame, face, "Unknown", "Người lạ", user_data.get('score', 0.0), is_known=False, status="FAILED", company_id=user_data.get('company_id'))
                            logger.warning(f"Unknown face 5 times. Cooldown 5s started.")
                        
                        elif f_data['unknown_attempts'] == 10:
                            # Sau 10 lần tổng cộng: Đánh dấu là Người lạ/Truy cập trái phép
                            f_data['status'] = 'UNAUTHORIZED'
                            self._save_log_with_bbox(frame, face, "Stranger", "Truy cập lạ", user_data.get('score', 0.0), is_known=False, status="FAILED", company_id=user_data.get('company_id'))
                            logger.error("Unauthorized access confirmed: 10 failed attempts.")
                        
                        else:
                            # Các lần thử khác chỉ log console để theo dõi
                            logger.warning(f"Unknown face. Attempt {f_data['unknown_attempts']}/10")
                
                updated_faces_map[matched_id] = f_data

            # Update visualization attributes
            if matched_id in updated_faces_map:
                data = updated_faces_map[matched_id]
                if data['user_data']:
                    u_d = data['user_data']
                    face.user_id = u_d.get('user_id', 'Unknown')
                    face.score = u_d.get('score', 0.0)
                    
                    if data['status'] == 'COOLDOWN':
                        name = remove_accents(u_d.get('name'))
                        face.name = f"{name} (Khoa {data['cooldown_remaining'] // 60}p)"
                    elif data['status'] == 'RETRY_WAIT':
                        wait_left = int(5 - (current_time - data['last_attempt_time']))
                        face.name = f"Chua ro (Thu lai {max(0, wait_left)}s)"
                    elif data['status'] == 'UNAUTHORIZED':
                        face.name = "!!! TRUY CAP LAI !!!"
                    else:
                        face.name = remove_accents(u_d.get('name', 'Chua ro'))
                else:
                    face.name = "Dang phan tich..."
                    face.score = 0.0

        self.active_faces = {k: v for k, v in updated_faces_map.items() if current_time - v['last_seen'] < 1.0}
