import cv2
import numpy as np
from loguru import logger
from typing import Tuple

class AntiSpoofing:
    """
    Simple Anti-Spoofing using Texture and Color Analysis (No Model Required).
    """
    
    def __init__(self, model_path: str = None, device_id: int = -1):
        """
        Initialize simple anti-spoofing logic.
        Model path is ignored in this version.
        """
        logger.success("Simple Anti-Spoofing (Non-AI Version) initialized.")

    def predict(self, frame: np.ndarray, bbox: np.ndarray) -> Tuple[int, float]:
        """
        Predict if a face is real or spoof using Laplacian Variance and Color Analysis.
        
        Returns:
            label: 1 for Real, 0 for Spoof
            score: Confidence value (0.0 to 1.0)
        """
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                return 1, 1.0
            
            # --- 1. Laplacian Variance (Blur/Detail Detection) ---
            # Real faces have natural texture. Photos/Screens are either too blurry or have artificial edges.
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Typical thresholds: Real faces are usually > 100. Prints are often < 60.
            # (Note: These thresholds might need tuning based on camera quality)
            # score_lap = min(1.0, lap_var / 150.0) 
            
            # --- 2. Color Space Analysis (YCrCb) ---
            # Human skin has a very specific distribution in Cr and Cb channels.
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)
            
            # Mean and Std Dev of Cr channel (Red-Diff)
            cr_mean = np.mean(cr)
            cr_std = np.std(cr)
            
            # Real skin: 133 < Cr < 173 and 77 < Cb < 127
            # Typically, spoofed images (screens/prints) have much higher or lower variance 
            # or shifted means due to lighting/reproduction.
            
            # Simple heuristic score
            is_skin_color = (130 < cr_mean < 180) and (cr_std > 5)
            
            # --- 3. Combined Logic ---
            # This is a basic approach. You can adjust the threshold '80' based on your camera.
            # High lap_var usually means a real, sharp face.
            is_sharp = lap_var > 70 
            
            if is_sharp and is_skin_color:
                label = 1 # Real
                confidence = min(1.0, lap_var / 250.0 + 0.3)
            else:
                label = 0 # Spoof
                confidence = 1.0 - (lap_var / 200.0)
                
            return label, float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            logger.debug(f"Anti-Spoofing error: {e}")
            return 1, 1.0 # Default to Real on error
