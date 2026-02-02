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
                allowed_modules=['detection', 'recognition'],
                providers=['CPUExecutionProvider'] # Forcing CPU as requested
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
            logger.success(f"InsightFace model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise e

    def detect_and_extract(self, frame: np.ndarray) -> List[Any]:
        """
        Detect faces and extract embeddings from a frame.
        
        Args:
            frame: Input image (BGR from OpenCV)
            
        Returns:
            List of Face objects containing bbox, kps, embedding, etc.
        """
        try:
            # InsightFace expects BGR image (OpenCV default)
            faces = self.app.get(frame)
            return faces
        except Exception as e:
            logger.error(f"Error during face detection/extraction: {e}")
            return []

    def draw_faces(self, frame: np.ndarray, faces: List[Any]) -> np.ndarray:
        """
        Draw bounding boxes and detailed metadata on the frame.
        """
        res_frame = frame.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            # Draw green rectangle for recognized faces, yellow for unknown
            is_known = getattr(face, 'name', 'Unknown') != "Unknown"
            color = (0, 255, 0) if is_known else (0, 255, 255)
            
            cv2.rectangle(res_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Display information
            if hasattr(face, 'name'):
                y_offset = bbox[1] - 10
                
                # Main Label: Name (Score)
                score = getattr(face, 'score', 0.0)
                if score is None: score = 0.0
                label = f"{face.name} ({score:.2f})"
                cv2.putText(res_frame, label, (bbox[0], y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if is_known:
                    # Metadata lines
                    meta_info = [
                        f"ID: {face.user_id}",
                        f"N-sinh: {getattr(face, 'birthday', 'N/A')}",
                        f"Gio: {getattr(face, 'detect_time', '')}",
                        f"Mau: {getattr(face, 'vector_count', 0)}"
                    ]
                    
                    for i, text in enumerate(meta_info):
                        # Draw below the box
                        cv2.putText(res_frame, text, (bbox[0], bbox[3] + 20 + (i * 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return res_frame


__all__ = ["FaceRecognition"]
