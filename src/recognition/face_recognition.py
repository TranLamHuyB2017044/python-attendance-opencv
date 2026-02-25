"""
Face Recognition module using InsightFace.
Handles face detection and feature extraction (embedding).
"""

import numpy as np
import cv2
from insightface.app import FaceAnalysis
from loguru import logger
from typing import List, Optional, Tuple, Any

from src.config import InsightFaceConfig, MODELS_DIR


class FaceRecognition:
    """
    Face Recognition handler using InsightFace (ArcFace).
    
    This class wraps InsightFace's FaceAnalysis to detect faces
    and generate 512-dimensional embeddings.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        det_size: Optional[Tuple[int, int]] = None,
        det_thresh: Optional[float] = None,
        ctx_id: int = -1,  # -1 for CPU, 0+ for GPU
    ):
        """
        Initialize InsightFace models.
        
        Args:
            model_name: Name of the model pack (e.g., 'buffalo_l', 'buffalo_s')
            det_size: Input size for face detection
            det_thresh: Threshold for face detection
            ctx_id: Context ID for execution (CPU/GPU)
        """
        self.model_name = model_name or InsightFaceConfig.MODEL_NAME
        self.det_size = det_size or InsightFaceConfig.DET_SIZE
        self.det_thresh = det_thresh or InsightFaceConfig.DET_THRESH
        
        try:
            # Initialize FaceAnalysis
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(MODELS_DIR),
                allowed_modules=['detection', 'landmark_2d_106'], # Removed 'recognition', 'attribute'
                providers=['CPUExecutionProvider'] # Forcing CPU as requested
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
            
            # Manual load of recognition model for the strict pipeline
            from insightface.model_zoo import get_model
            # Look for the .onnx file in the recognition folder of buffalo_l/buffalo_s
            rec_model_name = "w600k_r50.onnx" if "buffalo_l" in self.model_name else "w600k_mbf.onnx"
            rec_model_path = MODELS_DIR / "models" / self.model_name / rec_model_name
            self.rec_model = get_model(str(rec_model_path), providers=['CPUExecutionProvider'])
            self.rec_model.prepare(ctx_id=ctx_id)
            logger.info("Recognition model loaded manually for strict pipeline.")
            # Anti-Spoofing Setup (Silent-Face-Anti-Spoofing)
            from src.recognition.antispoofing import AntiSpoofing
            self.anti_spoof = AntiSpoofing()
            logger.info("Silent-Face-Anti-Spoofing (MiniFASNet) initialized.")
                
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise e

    def detect_and_extract(self, frame: np.ndarray, max_faces: Optional[int] = None, fast: bool = False) -> List[Any]:
        """
        Architecture:
        1. InsightFace Detection -> 2. SFAS Anti-spoofing -> 3. InsightFace Recognition
        
        If fast=True, skips SFAS and Recognition (Defer to tracker).
        """
        try:
            # 1. Detection & Landmarks only
            faces = self.app.get(frame)
            
            if not faces:
                return []

            # Sort by size
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            if max_faces is not None:
                faces = faces[:max_faces]

            if fast:
                # In fast mode, we just return the detections with placeholders
                for face in faces:
                    face.is_real = True # Default to True so it enters tracker's stabilizing
                    face.as_label = 2    # 2 = WAITING for analysis
                    face.as_score = 0.0
                return faces

            real_faces = []
            for face in faces:
                # 2. Anti-Spoofing (SFAS)
                if self.anti_spoof:
                    as_label, as_score = self.anti_spoof.predict(frame, face)
                    face.as_label = int(as_label)
                    face.as_score = float(as_score)
                    face.is_real = (face.as_label == 1)
                else:
                    face.is_real = True
                    face.as_score = 1.0

                # 3. Recognition (Only if Real)
                if face.is_real:
                    self.rec_model.get(frame, face)
                
                real_faces.append(face)

            return real_faces
            
        except Exception as e:
            logger.error(f"Error during face detection/extraction: {e}")
            return []

    def draw_faces(self, frame: np.ndarray, faces: List[Any]) -> np.ndarray:
        """
        Draw bounding boxes and detailed metadata with plain text (no shadow).
        """
        res_frame = frame.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            is_known = getattr(face, 'name', 'Unknown') != "Unknown"
            is_real = getattr(face, 'is_real', True)
            as_label = getattr(face, 'as_label', 1) # 2 = WAITING
            
            # Color logic: Red if Fake or Spoof Warning, Green if Known, Yellow if Unknown
            # Check for "CANH BAO", "SPOOF", or "TRUY CAP" in name/label to handle flicker
            display_name = getattr(face, 'name', '')
            if not is_real or any(kw in display_name for kw in ["CANH BAO", "SPOOF", "TRUY CAP"]):
                color = (0, 0, 255) # RED for spoof/unauthorized
            elif as_label == 2:
                color = (0, 255, 255) # Yellow while waiting
            else:
                color = (0, 255, 0) if is_known else (0, 255, 255)
            
            # Draw Box
            cv2.rectangle(res_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Main Label: Real/Fake status
            as_score = getattr(face, 'as_score', 0.0)
            if as_label == 2:
                liveness_label = "ANALYZING LIVENESS..."
            elif not is_real:
                liveness_label = f"GIA MAO (FAKE) | Score: {as_score:.2f}"
            else:
                liveness_label = f"REAL | Score: {as_score:.2f}"
            
            y_offset = bbox[1] - 10
            
            cv2.putText(res_frame, liveness_label, (bbox[0], y_offset - 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if hasattr(face, 'name'):
                # Main Label: Name (Score)
                score = getattr(face, 'score', 0.0) or 0.0
                label = f"{face.name} ({score:.2f})"
                cv2.putText(res_frame, label, (bbox[0], y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                
                # Second Label: Camera Name & Time
                from src.config import CameraConfig
                import time
                cam_time_label = f"{CameraConfig.CAMERA_NAME} | {time.strftime('%H:%M:%S')}"
                cv2.putText(res_frame, cam_time_label, (bbox[0], y_offset - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
                if is_known and is_real:
                    gender_map = {0: "Nu", 1: "Nam"}
                    gender_val = "N/A"
                    if hasattr(face, 'gender') and face.gender is not None:
                        gender_val = gender_map.get(int(face.gender), "N/A")
                        
                    age_val = "N/A"
                    if hasattr(face, 'age') and face.age is not None:
                        age_val = int(face.age)

                    meta_info = [
                        f"ID: {getattr(face, 'user_id', 'Unknown') or 'Unknown'}",
                        f"N-sinh: {getattr(face, 'birthday', 'N/A') or 'N/A'}",
                        f"Gio: {getattr(face, 'detect_time', 'N/A') or 'N/A'}",
                        f"Mau: {getattr(face, 'vector_count', 0) if getattr(face, 'vector_count', None) is not None else 0}"
                    ]
                    
                    for i, text in enumerate(meta_info):
                        pos = (bbox[0], bbox[3] + 25 + (i * 22))
                        cv2.putText(res_frame, text, pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            # If SPOOF or UNAUTHORIZED, add a big warning
            if color == (0, 0, 255):
                warning_msg = "WARNING: SECURITY ALERT!"
                if "CANH BAO" in display_name or "SPOOF" in display_name:
                    warning_msg = "WARNING: ANTI-SPOOFING TRIGGERED!"
                elif "TRUY CAP" in display_name:
                    warning_msg = "WARNING: UNAUTHORIZED ACCESS!"
                    
                cv2.putText(res_frame, warning_msg, (bbox[0], bbox[3] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        return res_frame
                
        return res_frame


__all__ = ["FaceRecognition"]
