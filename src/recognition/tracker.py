import time
import numpy as np
from loguru import logger
from typing import List, Dict, Any
import cv2
from src.config import RecognitionConfig, CAPTURES_DIR, ApiConfig
from src.utils.voice_manager import voice_mgr
from src.attendance.mongodb_mgr import mongo_db

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
                
                # Check delay between attempts (5 seconds)
                wait_time = current_time - f_data['last_attempt_time']
                can_attempt = wait_time >= 5.0

                # Trigger recognition if stabilized and not currently in a "Locked/Wait" state
                if f_data['status'] in ['STABILIZING', 'RETRY_WAIT'] and time_stayed >= self.threshold_seconds and can_attempt:
                    user_data = attendance_mgr.recognize(face.normed_embedding)
                    user_id = user_data.get('user_id')
                    user_name = user_data.get('name', 'Unknown')
                    
                    if user_name != "Unknown":
                        # CASE: SUCCESS
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
                                # Capture and log
                                img_url = None
                                if frame is not None:
                                    img_name = f"{user_id}_{int(current_time)}.jpg"
                                    img_path = str(CAPTURES_DIR / img_name)
                                    cv2.imwrite(img_path, frame)
                                    img_url = f"{ApiConfig.BASE_URL}/captures/{img_name}"

                                # Log to Cloud (Main)
                                status = mongo_db.log_attendance(user_id, user_name, frame=frame)
                                if status == 'IN':
                                    voice_mgr.speak(f"{user_name} đã vào")
                                elif status == 'OUT':
                                    voice_mgr.speak(f"{user_name} đã ra")
                                else:
                                    voice_mgr.speak(f"Xin cảm ơn {user_name}")
                        else:
                            f_data['status'] = 'RECOGNIZED'
                            f_data['user_data'] = user_data
                            self.user_cooldowns[user_id] = current_time
                            # Capture and log
                            img_url = None
                            if frame is not None:
                                img_name = f"{user_id}_{int(current_time)}.jpg"
                                img_path = str(CAPTURES_DIR / img_name)
                                cv2.imwrite(img_path, frame)
                                img_url = f"{ApiConfig.BASE_URL}/captures/{img_name}"
                                
                            status = mongo_db.log_attendance(user_id, user_name, frame=frame)
                            if status == 'IN':
                                voice_mgr.speak(f"{user_name} đã vào")
                            elif status == 'OUT':
                                voice_mgr.speak(f"{user_name} đã ra")
                            else:
                                voice_mgr.speak(f"Xin cảm ơn {user_name}")
                        
                        f_data['unknown_attempts'] = 0 # Reset on success
                    
                    else:
                        # CASE: UNKNOWN
                        f_data['unknown_attempts'] += 1
                        f_data['last_attempt_time'] = current_time
                        f_data['user_data'] = user_data
                        
                        # Capture and log failure
                        img_url = None
                        if frame is not None:
                            img_name = f"unknown_{int(current_time)}.jpg"
                            img_path = str(CAPTURES_DIR / img_name)
                            cv2.imwrite(img_path, frame)
                            img_url = f"{ApiConfig.BASE_URL}/captures/{img_name}"
                        
                        mongo_db.log_attendance("Unknown", "Người lạ", status="FAILED", frame=frame)

                        if f_data['unknown_attempts'] < 3:
                            f_data['status'] = 'RETRY_WAIT'
                            voice_mgr.speak("Xin vui lòng thử lại")
                            logger.warning(f"Unknown face. Attempt {f_data['unknown_attempts']}/3")
                        else:
                            f_data['status'] = 'UNAUTHORIZED'
                            voice_mgr.speak("Có người lạ truy cập trái phép")
                            logger.error("Unauthorized access detected: Multiple unknown attempts.")
                
                updated_faces_map[matched_id] = f_data

            # Update visualization attributes
            if matched_id in updated_faces_map:
                data = updated_faces_map[matched_id]
                if data['user_data']:
                    u_d = data['user_data']
                    face.user_id = u_d.get('user_id', 'Unknown')
                    face.score = u_d.get('score', 0.0)
                    
                    if data['status'] == 'COOLDOWN':
                        face.name = f"{u_d.get('name')} (Khoa {data['cooldown_remaining'] // 60}p)"
                    elif data['status'] == 'RETRY_WAIT':
                        wait_left = int(5 - (current_time - data['last_attempt_time']))
                        face.name = f"Chua ro (Thu lai {max(0, wait_left)}s)"
                    elif data['status'] == 'UNAUTHORIZED':
                        face.name = "!!! TRUY CAP LAI !!!"
                    else:
                        face.name = u_d.get('name', 'Chua ro')
                else:
                    face.name = "Dang phan tich..."
                    face.score = 0.0

        self.active_faces = {k: v for k, v in updated_faces_map.items() if current_time - v['last_seen'] < 1.0}
