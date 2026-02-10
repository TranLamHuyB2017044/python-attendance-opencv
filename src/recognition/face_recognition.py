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
            # It will download models automatically to ~/.insightface/models/ if not present
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(MODELS_DIR),
                allowed_modules=['detection', 'recognition', 'attribute'],
                providers=['CPUExecutionProvider'] # Forcing CPU as requested
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
            logger.success(f"InsightFace model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise e

    def detect_and_extract(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[Any]:
        """
        Detect faces and extract embeddings from a frame.
        Sorted by size (largest first).
        """
        try:
            faces = self.app.get(frame)
            
            # Sort by bbox area: (x2-x1)*(y2-y1) - largest first
            if faces:
                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                
                # Limit number of faces if specified
                if max_faces is not None:
                    faces = faces[:max_faces]
                    
            return faces
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
            color = (0, 255, 0) if is_known else (0, 255, 255)
            
            # Draw Box
            cv2.rectangle(res_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            if hasattr(face, 'name'):
                # Main Label: Name (Score)
                y_offset = bbox[1] - 10
                score = getattr(face, 'score', 0.0) or 0.0
                label = f"{face.name} ({score:.2f})"
                cv2.putText(res_frame, label, (bbox[0], y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                
                # Second Label: Camera Name & Time (displayed ABOVE the name to avoid overlap)
                from src.config import CameraConfig
                import time
                cam_time_label = f"{CameraConfig.CAMERA_NAME} | {time.strftime('%H:%M:%S')}"
                # Move it up by 20 pixels instead of down
                cv2.putText(res_frame, cam_time_label, (bbox[0], y_offset - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
                if is_known:
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
                        f"G-tinh: {gender_val} ({age_val}t)",
                        f"Gio: {getattr(face, 'detect_time', 'N/A') or 'N/A'}",
                        f"Mau: {getattr(face, 'vector_count', 0) if getattr(face, 'vector_count', None) is not None else 0}"
                    ]
                    
                    for i, text in enumerate(meta_info):
                        pos = (bbox[0], bbox[3] + 25 + (i * 22))
                        cv2.putText(res_frame, text, pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                
        return res_frame
                
        return res_frame


__all__ = ["FaceRecognition"]
