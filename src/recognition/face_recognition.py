"""
Face Recognition module using InsightFace.
Handles face detection and feature extraction (embedding).
"""

import numpy as np
import cv2
from insightface.app import FaceAnalysis
from loguru import logger
from typing import List, Optional, Tuple, Any

from src.config import InsightFaceConfig, MODELS_DIR, RecognitionConfig


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
            # Initialize FaceAnalysis — detection + landmark only
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(MODELS_DIR),
                allowed_modules=['detection', 'landmark_2d_106'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
            logger.info(f"InsightFace detection model ready. DET_SIZE={self.det_size}")

            # Manual load of recognition (ArcFace embedding) model
            from insightface.model_zoo import get_model
            rec_model_name = "w600k_r50.onnx" if "buffalo_l" in self.model_name else "w600k_mbf.onnx"
            rec_model_path = MODELS_DIR / "models" / self.model_name / rec_model_name
            self.rec_model = get_model(str(rec_model_path), providers=['CPUExecutionProvider'])
            self.rec_model.prepare(ctx_id=ctx_id)
            logger.info("ArcFace recognition model loaded.")

            # Anti-Spoofing — chỉ load khi ANTI_SPOOFING_ENABLED=true
            if RecognitionConfig.ANTI_SPOOFING_ENABLED:
                from src.recognition.antispoofing import AntiSpoofing
                self.anti_spoof = AntiSpoofing()
                logger.info("[Anti-Spoofing] MiniFASNet loaded (ENABLED).")
            else:
                self.anti_spoof = None
                logger.warning("[Anti-Spoofing] DISABLED — skipping model load. All faces treated as REAL.")

        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise e

    def detect_and_extract(self, frame: np.ndarray, max_faces: Optional[int] = None,
                           fast: bool = False, roi=None) -> List[Any]:
        """
        Architecture:
        1. [Optional ROI crop] - chỉ detect trên vùng quan tâm → nhanh hơn
        2. InsightFace Detection (SCRFD)
        3. InsightFace Recognition (ArcFace) — defer to tracker nếu fast=True

        roi: (x1, y1, x2, y2) tọa độ full frame.
             Nếu set → detect chỉ trên vùng đó, sau đó remap bbox về tọa độ gốc.
        """
        try:
            # ─── ROI Crop: giảm pixels → SCRFD nhanh hơn ──────────────────
            offset_x, offset_y = 0, 0
            detect_frame = frame

            if roi is not None:
                fx1, fy1, fx2, fy2 = roi
                fh, fw = frame.shape[:2]
                fx1 = max(0, fx1); fy1 = max(0, fy1)
                fx2 = min(fw, fx2); fy2 = min(fh, fy2)
                if fx2 > fx1 and fy2 > fy1:
                    detect_frame = frame[fy1:fy2, fx1:fx2]
                    offset_x, offset_y = fx1, fy1

            # 1. Detection & Landmarks
            faces = self.app.get(detect_frame)

            # Remap bbox về tọa độ full frame nếu dùng ROI
            if offset_x != 0 or offset_y != 0:
                for face in faces:
                    face.bbox[0] += offset_x; face.bbox[2] += offset_x
                    face.bbox[1] += offset_y; face.bbox[3] += offset_y
                    if hasattr(face, 'kps') and face.kps is not None:
                        face.kps[:, 0] += offset_x
                        face.kps[:, 1] += offset_y

            if not faces:
                return []

            # Filter out faces that are too small or too far away
            valid_faces = []
            min_size = getattr(RecognitionConfig, "MIN_FACE_SIZE", 80)
            for face in faces:
                w = face.bbox[2] - face.bbox[0]
                h = face.bbox[3] - face.bbox[1]
                if w >= min_size and h >= min_size:
                    valid_faces.append(face)
            faces = valid_faces
            
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
                # 2. Anti-Spoofing — chỉ chạy nếu được bật VÀ model đã load
                if self.anti_spoof is not None and RecognitionConfig.ANTI_SPOOFING_ENABLED:
                    as_label, as_score = self.anti_spoof.predict(frame, face)
                    face.as_label = int(as_label)
                    face.as_score = float(as_score)
                    face.is_real = (face.as_label == 1)
                else:
                    # Anti-spoofing tắt: mặc định là người thật, as_label=1
                    face.is_real = True
                    face.as_label = 1
                    face.as_score = 1.0

                # 3. Recognition (ArcFace embedding)
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
            display_name = getattr(face, 'name', '') or ''

            # ╔════════════════════════════════════╗
            # ║ COLOR LOGIC (theo trạng thái tracker)        ║
            # ║  is_recognized flag do tracker gán trực tiếp  ║
            # ╚════════════════════════════════════╝
            is_recognized = getattr(face, 'recognized', False)
            is_spoof      = getattr(face, 'is_spoof',   False)

            if is_spoof or any(kw in str(display_name) for kw in ["TRUY CAP", "GIA MAO"]):
                color = (0, 0, 255)       # 🔴 Đỏ: spoof / unauthorized
            elif is_recognized:
                color = (0, 220, 0)       # 🟢 Xanh: đã nhận diện thành công
            else:
                color = (0, 200, 255)     # 🟡 Vàng: đang phân tích / chưa rõ


            # Box
            cv2.rectangle(res_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            y_offset = bbox[1] - 10

            if hasattr(face, 'name'):
                score = getattr(face, 'score', 0.0) or 0.0
                label = f"{face.name} ({score:.2f})" if is_recognized and score > 0 else face.name
                cv2.putText(res_frame, label, (bbox[0], y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                from src.config import CameraConfig
                import time
                cam_time_label = f"{CameraConfig.CAMERA_NAME} | {time.strftime('%H:%M:%S')}"
                cv2.putText(res_frame, cam_time_label, (bbox[0], y_offset - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                if is_recognized:
                    meta_info = [
                        f"ID: {getattr(face, 'user_id', 'Unknown') or 'Unknown'}",
                        f"N-sinh: {getattr(face, 'birthday', 'N/A') or 'N/A'}",
                        f"Gio: {getattr(face, 'detect_time', 'N/A') or 'N/A'}",
                        f"Mau: {getattr(face, 'vector_count', 0) if getattr(face, 'vector_count', None) is not None else 0}"
                    ]
                    for i, text in enumerate(meta_info):
                        cv2.putText(res_frame, text, (bbox[0], bbox[3] + 25 + i * 22),
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
