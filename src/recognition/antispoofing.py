import cv2
import numpy as np
import time
from loguru import logger
from typing import Tuple, Any

class AntiSpoofing:
    """
    Advanced Anti-Spoofing using Blink Detection (Liveness) and Texture Analysis.
    """
    
    def __init__(self, model_path: str = None, device_id: int = -1):
        # {face_index: {'bbox': bbox, 'ear_history': [], 'blink_count': 0, 'is_real': False, 'last_seen': time}}
        self.history = {}
        self.max_history = 10
        
        # EAR thresholds
        self.EAR_THRESHOLD = 0.22  # Below this, eye is considered closed
        self.BLINK_FRAME_MIN = 1   # Minimum frames closed
        self.BLINK_FRAME_MAX = 5   # Maximum frames closed (avoid slow blinks/static photos)
        
        logger.success("Blink-aware Anti-Spoofing initialized.")

    def _get_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def calculate_ear(self, landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) using InsightFace 106 landmarks.
        Left eye: 35(I), 39(O), 37(T1), 38(T2), 41(B1), 40(B2)
        Right eye: 89(I), 93(O), 91(T1), 92(T2), 95(B1), 94(B2)
        """
        def eye_aspect_ratio(eye_pts):
            # Vertical distances
            v1 = np.linalg.norm(eye_pts[2] - eye_pts[5]) # 37-41
            v2 = np.linalg.norm(eye_pts[3] - eye_pts[4]) # 38-40
            # Horizontal distance
            h = np.linalg.norm(eye_pts[0] - eye_pts[1])  # 35-39
            return (v1 + v2) / (2.0 * h)

        # Extract eye points
        left_eye = landmarks[[35, 39, 37, 38, 40, 41]]
        right_eye = landmarks[[89, 93, 91, 92, 94, 95]]
        
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        
        return (ear_left + ear_right) / 2.0

    def predict(self, frame: np.ndarray, face_obj: Any) -> Tuple[int, float]:
        """
        Main anti-spoofing logic combining Blink Detection and Texture Analysis.
        """
        try:
            bbox = face_obj.bbox
            landmarks = getattr(face_obj, 'landmark_2d_106', None)
            
            if landmarks is None:
                return 1, 0.5 # Fallback if no landmarks
                
            # --- 1. Texture/Glare Detection (Passive) ---
            # Phone screens still produce glare and unnatural texture
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_roi = gray[y1:y2, x1:x2]
            
            # Glare check
            _, bright = cv2.threshold(face_roi, 240, 255, cv2.THRESH_BINARY)
            glare_ratio = np.sum(bright == 255) / float(face_roi.size)
            
            # --- 2. Blink Detection (Active Liveness) ---
            ear = self.calculate_ear(landmarks)
            
            # Match with history
            best_match_id = None
            for fid, h_data in list(self.history.items()):
                if self._get_iou(bbox, h_data['bbox']) > 0.6:
                    best_match_id = fid
                    break
            
            is_blinked = False
            if best_match_id is not None:
                h = self.history[best_match_id]
                h['bbox'] = bbox
                h['last_seen'] = time.time()
                
                # Check for state transition (Open -> Closed -> Open)
                # h['ear_state']: 0=open, 1=closed
                prev_state = h.get('ear_state', 0)
                current_state = 1 if ear < self.EAR_THRESHOLD else 0
                
                if prev_state == 0 and current_state == 1:
                    # Closing
                    h['closed_start_time'] = time.time()
                    h['ear_state'] = 1
                elif prev_state == 1 and current_state == 0:
                    # Opening
                    duration = time.time() - h.get('closed_start_time', 0)
                    # A blink usually lasts 0.1 to 0.4 seconds
                    if 0.05 < duration < 0.5:
                        h['blink_count'] += 1
                        logger.info(f"Blink detected! Total: {h['blink_count']}")
                        if h['blink_count'] >= 1:
                            h['is_real'] = True
                    h['ear_state'] = 0
                elif current_state == 1:
                    # Stay closed too long? (Likely static photo or closed eyes)
                    duration = time.time() - h.get('closed_start_time', 0)
                    if duration > 1.0:
                        h['is_real'] = False # Reset if eyes closed too long
                
                is_blinked = h['is_real']
            else:
                new_id = int(time.time() * 1000)
                self.history[new_id] = {
                    'bbox': bbox, 
                    'ear_state': 0, 
                    'blink_count': 0, 
                    'is_real': False, 
                    'last_seen': time.time()
                }
                # Cleanup old history
                if len(self.history) > 10:
                    oldest = min(self.history.keys(), key=lambda k: self.history[k]['last_seen'])
                    self.history.pop(oldest)

            # --- 3. Final Decision ---
            # Level 1: Screen Glare = 100% SPOOF
            if glare_ratio > 0.05:
                label = 0 # Spoof
                confidence = 0.95
            # Level 2: Blink Detected = 100% REAL
            elif is_blinked:
                label = 1 # Real
                confidence = 0.99
            # Level 3: No blink yet = WAITING
            else:
                label = 2 # WAITING (New state)
                confidence = 0.5
                
            return label, float(confidence)

        except Exception as e:
            logger.debug(f"Anti-Spoofing error: {e}")
            return 1, 1.0
